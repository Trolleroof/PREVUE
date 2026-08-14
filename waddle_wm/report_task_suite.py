"""Score the task-suite world model as a verifier, against controls that can actually beat it.

The question is never "is the accuracy high". On a corpus with a 60/40 label split a constant
answer scores 0.60, and a model given the block coordinates can compute most of a pick-and-
place outcome with arithmetic. The question is whether *pixels* buy anything, so every number
here is reported beside the strongest thing that could produce it without pixels:

| control | what it has | what it cannot know |
| --- | --- | --- |
| majority class | nothing | everything |
| geometry rule | the plan, as thresholds | anything not in the plan |
| plan-only forest | the plan, the task, every block's xyz | each block's *heading* |
| no-vision ablation | the same, through the same network | the same |
| **+ oracle heading** | all of the above *and* the true headings | — it is the ceiling |
| world model | all of the above, and the observation window | — |

The oracle-heading row is the diagnostic that makes the result readable. The corpus is built so
the heading decides the grasp and the coordinates omit it, so the gap between the plan-only
forest and the oracle-heading forest is the value of knowing the headings. How much of that gap
the world model closes is how much of the heading it actually read out of the pixels.

The `vision_decides` slice is the same argument per episode: plans that are well aimed in
coordinates but commanded across the block. They look like good plans to everything without
pixels, and they fail.

    uv run python -m waddle_wm.report_task_suite --data data/ur5e_wm_suite \\
        --checkpoint models/task_suite_world_model.pt --out results/task_suite_world_model.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from waddle_wm import plan_encoding
from waddle_wm.sim.generate_suite import FAMILIES
from waddle_wm.sim.validate_suite import misalignment_deg
from waddle_wm.train_task_suite_world_model import (SUBTASK_SLOTS, SuiteWorldModel,
                                                    apply_context_projection, apply_normaliser,
                                                    assemble, ensemble_scores)

WELL_AIMED_GRASP_M = 0.015      # the plan aims at the block, in coordinates
WELL_AIMED_PLACE_M = 0.045      # and at the destination
ALIGNED_DEG, ACROSS_DEG = 20.0, 40.0


def load_members(checkpoint: dict, device, key: str = "members") -> list[SuiteWorldModel]:
    members = []
    for state in checkpoint.get(key) or []:
        model = SuiteWorldModel(checkpoint["context_dim"], checkpoint["plan_dim"],
                                checkpoint["task_dim"], hidden=checkpoint["hidden"],
                                context_width=checkpoint["context_width"],
                                dropout=checkpoint["dropout"]).to(device)
        model.load_state_dict(state)
        model.eval()
        members.append(model)
    return members


def verdict_metrics(probability: np.ndarray, truth: np.ndarray, threshold: float,
                    uncertainty: np.ndarray | None = None) -> dict:
    verdict = (probability >= threshold).astype(float)
    negatives, positives = max(1, int((truth == 0).sum())), max(1, int((truth == 1).sum()))
    correct = verdict == truth
    metrics = {
        "episodes": int(len(truth)),
        "accuracy": float(correct.mean()),
        "brier": float(((probability - truth) ** 2).mean()),
        "false_accept_rate": float(((verdict == 1) & (truth == 0)).sum() / negatives),
        "false_reject_rate": float(((verdict == 0) & (truth == 1)).sum() / positives),
        "base_rate": float(truth.mean()),
    }
    if uncertainty is not None:
        metrics["uncertainty_correct"] = float(uncertainty[correct].mean()) if correct.any() else 0.0
        metrics["uncertainty_wrong"] = float(uncertainty[~correct].mean()) if (~correct).any() else 0.0
        metrics["uncertainty_ratio"] = float(metrics["uncertainty_wrong"] /
                                             max(metrics["uncertainty_correct"], 1e-9))
    return metrics


def episode_geometry(record: dict, raw: dict, index: int) -> dict:
    """Per-episode quantities the slices are cut on, all of them plan-time except the heading."""
    plans, mask = raw["plan"][index], raw["mask"][index]
    grasp = max(float(np.linalg.norm(plans[k][:2])) for k in range(SUBTASK_SLOTS) if mask[k])
    place = max(float(np.linalg.norm(plans[k][3:5])) for k in range(SUBTASK_SLOTS) if mask[k])
    misalignment = max(misalignment_deg(s) for s in record["skill"]["params"]["subtasks"])
    return {"grasp_offset_m": grasp, "place_offset_m": place, "misalignment_deg": misalignment}


def slice_masks(records: list[dict], raw: dict) -> dict[str, np.ndarray]:
    """The named groups the headline table is decomposed over."""
    geometry = [episode_geometry(record, raw, index) for index, record in enumerate(records)]
    grasp = np.array([g["grasp_offset_m"] for g in geometry])
    place = np.array([g["place_offset_m"] for g in geometry])
    misalignment = np.array([g["misalignment_deg"] for g in geometry])

    well_aimed = (grasp <= WELL_AIMED_GRASP_M) & (place <= WELL_AIMED_PLACE_M)
    aligned, across = misalignment <= ALIGNED_DEG, misalignment >= ACROSS_DEG
    groups = {
        "plan_decides": place > WELL_AIMED_PLACE_M,
        "well_aimed_and_aligned": well_aimed & aligned,
        "vision_decides": well_aimed & across,
        # The one slice that cannot be gamed. `vision_decides` is almost all failures, so
        # refusing everything scores 1.000 on it; `well_aimed_and_aligned` is mostly successes,
        # so accepting everything scores well there. Their union contains both, and across it
        # the coordinates are drawn from the same distribution — the *only* systematic
        # difference is a heading that exists in the pixels and nowhere else. Accuracy here is
        # the claim: an arm without pixels cannot do better than the base rate.
        "orientation_discrimination": well_aimed & (aligned | across),
    }
    for low, high in ((0, 10), (10, 40), (40, 60), (60, 91)):
        groups[f"misalignment_{low}_{high}deg"] = (misalignment >= low) & (misalignment < high)
    for family in FAMILIES:
        groups[f"family_{family}"] = raw["families"] == family
    return groups


def control_features(raw: dict, headings: np.ndarray | None = None) -> np.ndarray:
    """Everything a control sees: the plan, the task, the coordinates — and no pixels."""
    columns = [raw["plan"].reshape(len(raw["plan"]), -1),
               raw["task"].reshape(len(raw["task"]), -1),
               raw["mask"], raw["initial"]]
    if headings is not None:
        columns.append(headings)
    return np.concatenate(columns, axis=1)


def true_headings(records: list[dict], block_names) -> np.ndarray:
    """sin 2y, cos 2y of every block's actual spawn heading: the oracle's extra knowledge."""
    rows = []
    for record in records:
        yaws = record["skill"]["params"]["block_yaws_deg"]
        row = []
        for name in block_names:
            angle = 2.0 * np.radians(float(yaws[name]))
            row.extend([np.sin(angle), np.cos(angle)])
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def forest(features: np.ndarray, truth: np.ndarray, train: np.ndarray, test: np.ndarray,
           seed: int = 0) -> np.ndarray:
    model = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    model.fit(features[train], truth[train])
    return model.predict_proba(features[test])[:, 1]


def rules_probability(raw: dict) -> np.ndarray:
    """A deterministic geometry rule with no image access: is every subtask well aimed?

    This is the arm of the earlier comparison that beat the visual world model, so it is the
    one to keep honest. It cannot see a heading, which on this corpus is the point.
    """
    scores = []
    for plans, mask in zip(raw["plan"], raw["mask"]):
        ok = 1.0
        for k in range(SUBTASK_SLOTS):
            if not mask[k]:
                continue
            grasp = float(np.linalg.norm(plans[k][:2]))
            place = float(np.linalg.norm(plans[k][3:5]))
            ok = min(ok, float(grasp <= WELL_AIMED_GRASP_M and place <= WELL_AIMED_PLACE_M))
        scores.append(ok)
    return np.asarray(scores, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_wm_suite"))
    ap.add_argument("--embeddings", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=Path("models/task_suite_world_model.pt"))
    ap.add_argument("--out", type=Path, default=Path("results/task_suite_world_model.json"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if saved.get("model_type") != "task_suite_state":
        raise SystemExit(f"{args.checkpoint} is a {saved.get('model_type')!r} checkpoint")
    plan_encoding.require_orientation_aware("the task-suite verifier", plan_encoding.declared(saved))

    manifest = json.loads((args.data / "manifest.json").read_text())
    records = [json.loads(line) for line in (args.data / "records.jsonl").open()]
    cache = torch.load(args.embeddings or args.data / "window_embeddings.pt", weights_only=False)
    raw = assemble(records, manifest, cache, saved["plan_encoding"]["version"])
    block_names = tuple(manifest["block_names"])

    truth = raw["success"]
    train = raw["splits"] == "train"
    evaluate = raw["splits"] == args.split
    threshold = float(saved["decision_threshold"])

    data = {key: torch.from_numpy(np.asarray(raw[key])).to(device)
            for key in ("initial", "final", "plan", "task", "mask",
                        "subtask_labels", "success")}
    data["context"] = apply_context_projection(torch.from_numpy(raw["context"]).to(device),
                                               saved.get("context_projection"))
    for key in ("context", "plan", "initial", "final"):
        data[key] = apply_normaliser(data[key], saved["normalisation"][key])

    members = load_members(saved, device)
    rows = torch.from_numpy(np.flatnonzero(evaluate)).to(device)
    probability, subtask_probability, predicted_state, uncertainty = ensemble_scores(members, data, rows)

    # The ablation is a *separately trained* no-pixels ensemble from the checkpoint. Falling back
    # to zeroing this model's context would compare it against an input it was never fitted on.
    blind_members = load_members(saved, device, "blind_members")
    if not blind_members:
        raise SystemExit("this checkpoint has no trained blind control; retrain to get one, "
                         "rather than zeroing the context of a model fitted with it")
    blind_data = {**data, "context": torch.zeros_like(data["context"])}
    blind_probability, _, _, _ = ensemble_scores(blind_members, blind_data, rows)
    probability = probability.cpu().numpy()
    uncertainty = uncertainty.cpu().numpy()
    blind_probability = blind_probability.cpu().numpy()

    # Controls. Each is fitted on train and scored on the evaluation split, same as the model.
    headings = true_headings(records, block_names)
    plan_only = forest(control_features(raw), truth, train, evaluate)
    oracle_heading = forest(control_features(raw, headings), truth, train, evaluate)
    rules = rules_probability(raw)[evaluate]
    majority = float(max(truth[evaluate].mean(), 1 - truth[evaluate].mean()))

    target = truth[evaluate]
    headline = {
        "majority_class": {"accuracy": majority, "episodes": int(evaluate.sum())},
        "geometry_rule": verdict_metrics(rules, target, 0.5),
        "plan_only_forest": verdict_metrics(plan_only, target, 0.5),
        "no_vision_ablation": verdict_metrics(blind_probability, target, threshold),
        "oracle_heading_forest": verdict_metrics(oracle_heading, target, 0.5),
        "world_model": verdict_metrics(probability, target, threshold, uncertainty),
    }

    # Per-subtask decisions, which is where a sequence model earns its keep.
    subtask_probability = subtask_probability.cpu().numpy()
    mask = raw["mask"][evaluate]
    labels = raw["subtask_labels"][evaluate]
    per_subtask = {}
    for axis, name in enumerate(("lifted", "placed", "success")):
        present = mask.reshape(-1) > 0
        predicted = (subtask_probability[..., axis].reshape(-1) >= 0.5)[present]
        actual = labels[..., axis].reshape(-1)[present]
        per_subtask[name] = {"n": int(present.sum()), "accuracy": float((predicted == actual).mean()),
                             "base_rate": float(actual.mean())}

    # State prediction: where the model thinks every block ends up, in metres.
    state_stats = saved["normalisation"]["final"]
    denorm = (predicted_state.cpu().numpy() * state_stats["std"]) + state_stats["mean"]
    actual_state = raw["final"][evaluate]
    block_error = np.linalg.norm((denorm[:, :9] - actual_state[:, :9]).reshape(-1, 3, 3), axis=2)

    groups = slice_masks(records, raw)
    slices = {}
    for name, group in groups.items():
        selected = group[evaluate]
        if selected.sum() < 5:
            continue
        def auc(scores):
            """Ranking quality, independent of where the threshold sits."""
            actual = target[selected]
            if actual.min() == actual.max():
                return None
            order = np.argsort(scores[selected], kind="mergesort")
            ranks = np.empty(len(order), dtype=float)
            ranks[order] = np.arange(1, len(order) + 1)
            positives, negatives = actual.sum(), (1 - actual).sum()
            return float((ranks[actual == 1].sum() - positives * (positives + 1) / 2) /
                         (positives * negatives))

        slices[name] = {
            "n": int(selected.sum()),
            "base_rate": float(target[selected].mean()),
            "auc_world_model": auc(probability),
            "auc_no_vision_ablation": auc(blind_probability),
            "auc_plan_only_forest": auc(plan_only),
            "auc_oracle_heading_forest": auc(oracle_heading),
            "world_model": float(((probability[selected] >= threshold) == target[selected]).mean()),
            "no_vision_ablation": float(((blind_probability[selected] >= threshold) == target[selected]).mean()),
            "plan_only_forest": float(((plan_only[selected] >= 0.5) == target[selected]).mean()),
            "oracle_heading_forest": float(((oracle_heading[selected] >= 0.5) == target[selected]).mean()),
            "geometry_rule": float(((rules[selected] >= 0.5) == target[selected]).mean()),
            "mean_p_success": float(probability[selected].mean()),
        }

    report = {
        "checkpoint": str(args.checkpoint), "data": str(args.data), "split": args.split,
        "episodes": int(evaluate.sum()), "decision_threshold": threshold,
        "plan_encoding": saved["plan_encoding"],
        "headline": headline,
        "per_subtask": per_subtask,
        "state_prediction": {"block_xy_rmse_m": float(np.sqrt(((denorm[:, :9] - actual_state[:, :9]) ** 2).mean())),
                             "block_position_median_error_m": float(np.median(block_error))},
        "slices": slices,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"\n{args.split} split, {int(evaluate.sum())} episodes, threshold {threshold:.3f}\n")
    print(f"{'arm':<24} {'acc':>7} {'brier':>7} {'false-acc':>10} {'false-rej':>10}")
    for name, values in headline.items():
        print(f"{name:<24} {values['accuracy']:>7.3f} {values.get('brier', float('nan')):>7.3f} "
              f"{values.get('false_accept_rate', float('nan')):>10.3f} "
              f"{values.get('false_reject_rate', float('nan')):>10.3f}")
    print(f"\naccuracy by slice")
    print(f"{'slice':<30} {'n':>5} {'base':>6} {'model':>7} {'blind':>7} {'plan':>7} {'oracle':>7}")
    for name, values in slices.items():
        print(f"{name:<30} {values['n']:>5} {values['base_rate']:>6.2f} {values['world_model']:>7.3f} "
              f"{values['no_vision_ablation']:>7.3f} {values['plan_only_forest']:>7.3f} "
              f"{values['oracle_heading_forest']:>7.3f}")
    print(f"\nAUC by slice (ranking quality, threshold-independent; '-' = one class only)")
    print(f"{'slice':<30} {'n':>5} {'model':>7} {'blind':>7} {'plan':>7} {'oracle':>7}")
    for name, values in slices.items():
        cells = [values[f"auc_{arm}"] for arm in
                 ("world_model", "no_vision_ablation", "plan_only_forest", "oracle_heading_forest")]
        rendered = "".join(f"{'      -' if c is None else f'{c:>7.3f}'}" for c in cells)
        print(f"{name:<30} {values['n']:>5}{rendered}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
