"""Fine-tune the yaw-aware visual model on paired candidate outcomes.

    uv run python -m waddle_wm.train_paired_candidate_ranker \
      --train data/counterfactual/train-diagnostic.json \
      --calibration data/counterfactual/calibration-diagnostic.json \
      --pools data/pools --checkpoint models/multiblock_world_model_yaw.pt \
      --out models/paired_candidate_ranker.pt

The locked test split is intentionally not accepted here.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from waddle_wm import benchmark_selectors as benchmark
from waddle_wm import selectors
from waddle_wm.train_multiblock_world_model import StateWorldModel, task_features


def paired_indices(labels: torch.Tensor, groups: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """All success/failure comparisons within one frozen observation."""
    positive, negative = [], []
    for group in sorted(set(groups)):
        members = [i for i, value in enumerate(groups) if value == group]
        yes = [i for i in members if labels[i] == 1]
        no = [i for i in members if labels[i] == 0]
        for left in yes:
            for right in no:
                positive.append(left); negative.append(right)
    if not positive:
        raise ValueError("fitting data contains no within-scene success/failure pairs")
    return torch.tensor(positive, device=labels.device), torch.tensor(negative, device=labels.device)


def fit_split(artifact: dict) -> str:
    split = (artifact.get("metadata", {}).get("dataset") or {}).get("split")
    if split not in ("train", "calibration"):
        raise ValueError(f"refusing to fit on {split!r}; only train/calibration are allowed")
    return split


def load_split(path: Path, pools_root: Path, arm: selectors.VisualWorldModel) -> dict:
    artifact = json.loads(path.read_text())
    split = fit_split(artifact)
    pools = benchmark.load_pools(pools_root, artifact)
    views = benchmark.load_views(path, pools)
    widest = {}
    for scene in artifact["scenes"]:
        if scene["physics_seed"] == 0 and scene["prefix"] > widest.get(scene["pool_id"], {}).get("prefix", 0):
            widest[scene["pool_id"]] = scene

    rows, labels, groups, slices = [], [], [], []
    for pool_id, scene in sorted(widest.items()):
        pool, view = pools[pool_id], views[pool_id]
        context = selectors.ScenarioContext(
            view, benchmark.observation_window(pool, arm.manifest["window_frames"]))
        arm.prepare(context)
        estimates, task = context.estimates(), view["task"]
        raw_state = arm._state(estimates)
        state = arm._normalise(raw_state, "state")
        task_row = task_features([task["object"].replace(" ", "_")],
                                 [task["destination"].replace(" ", "_")],
                                 arm.block_names, arm.device)
        candidates = {candidate["candidate_id"]: candidate for candidate in view["candidates"]}
        for candidate_id in scene["pool_prefix"]:
            candidate = candidates[candidate_id]
            plan = None if candidate.get("aborts") else arm._plan(candidate, estimates, task, raw_state)
            if plan is None:
                continue
            rows.append((arm._latent.squeeze(0).cpu(), state.squeeze(0).cpu(),
                         arm._normalise(plan, "plan").squeeze(0).cpu(), task_row.squeeze(0).cpu()))
            labels.append(float(scene["counterfactual"][candidate_id]["success"]))
            groups.append(pool_id)
            slices.append((pool["scene"].get("suite") or {}).get("outcome_slice", "unsliced"))
    if len(set(groups)) != len(widest):
        raise ValueError("paired rows do not preserve one group per frozen pool")
    tensors = tuple(torch.stack([row[index] for row in rows]) for index in range(4))
    target = torch.tensor(labels)
    pairs = paired_indices(target, groups)
    return {"inputs": tensors, "labels": target, "groups": groups, "slices": slices,
            "pairs": pairs, "split": split, "pools": len(widest)}


def logits(members: nn.ModuleList, data: dict, use_context: bool = True) -> torch.Tensor:
    context, state, plan, task = data["inputs"]
    if not use_context:
        context = torch.zeros_like(context)
    return torch.stack([member(context, state, plan, task)[1][:, 1] for member in members])


def loss(members: nn.ModuleList, data: dict) -> torch.Tensor:
    scores = logits(members, data)
    target = data["labels"].expand_as(scores)
    positive, negative = data["pairs"]
    return (nn.functional.binary_cross_entropy_with_logits(scores, target)
            + nn.functional.softplus(-(scores[:, positive] - scores[:, negative])).mean())


def metrics(members: nn.ModuleList, data: dict, threshold: float = 0.5,
            use_context: bool = True) -> dict:
    with torch.inference_mode():
        probability = logits(members, data, use_context).sigmoid().mean(0)
    labels, groups = data["labels"], data["groups"]
    positive, negative = data["pairs"]
    verdict = probability >= threshold
    selected = []
    for group in sorted(set(groups)):
        indices = [i for i, value in enumerate(groups) if value == group]
        chosen = max(indices, key=lambda i: float(probability[i]))
        selected.append(float(labels[chosen]))
    failures, successes = labels == 0, labels == 1
    return {"candidates": len(labels), "pools": len(set(groups)),
            "pair_accuracy": float((probability[positive] > probability[negative]).float().mean()),
            "selected_success": float(np.mean(selected)),
            "brier": float((probability - labels).square().mean()),
            "false_accept_rate": float((verdict & failures).sum() / failures.sum().clamp_min(1)),
            "false_reject_rate": float(((~verdict) & successes).sum() / successes.sum().clamp_min(1)),
            "threshold": threshold}


def subset(data: dict, slice_name: str) -> dict:
    index = torch.tensor([i for i, value in enumerate(data["slices"]) if value == slice_name],
                         device=data["labels"].device)
    labels = data["labels"][index]
    groups = [data["groups"][i] for i in index.cpu().tolist()]
    return {**data, "inputs": tuple(value[index] for value in data["inputs"]),
            "labels": labels, "groups": groups,
            "slices": [slice_name] * len(index), "pairs": paired_indices(labels, groups)}


def sliced_metrics(members: nn.ModuleList, data: dict, threshold: float) -> dict:
    return {name: metrics(members, subset(data, name), threshold)
            for name in sorted(set(data["slices"]))}


def move(data: dict, device) -> dict:
    return {**data, "inputs": tuple(value.to(device) for value in data["inputs"]),
            "labels": data["labels"].to(device),
            "pairs": tuple(value.to(device) for value in data["pairs"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--pools", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--encoder", type=Path, default=Path("models/vjepa2-vitl-fpc64-256"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    arm = selectors.VisualWorldModel(args.checkpoint, args.encoder, device=device)
    train = move(load_split(args.train, args.pools, arm), device)
    calibration = move(load_split(args.calibration, args.pools, arm), device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    members = nn.ModuleList([StateWorldModel(saved["context_dim"], saved["plan_dim"])
                             for _ in range(saved["member_count"])])
    members.load_state_dict(saved["members"]); members.to(device)
    baseline = {"visual": metrics(members, calibration),
                "without_context": metrics(members, calibration, use_context=False)}
    optimizer = torch.optim.AdamW(members.parameters(), lr=args.lr, weight_decay=1e-4)
    best = (float("inf"), 0, None)
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(); loss(members, train).backward(); optimizer.step()
        with torch.inference_mode():
            value = float(loss(members, calibration))
        if value < best[0]:
            best = (value, epoch, deepcopy(members.state_dict()))
        if epoch % 50 == 0:
            print(f"epoch {epoch}: calibration loss {value:.4f} (best {best[0]:.4f} @ {best[1]})", flush=True)
    members.load_state_dict(best[2]); members.eval()

    with torch.inference_mode():
        probability = logits(members, calibration).sigmoid().mean(0)
    candidates = torch.unique(probability).sort().values.tolist()
    labels = calibration["labels"]
    safe = [value for value in candidates if float(
        (((probability >= value) & (labels == 0)).sum() / (labels == 0).sum().clamp_min(1))) <= 0.10]
    threshold = max(safe, key=lambda value: float(((probability >= value) == labels.bool()).float().mean())) \
        if safe else 1.0
    report = {"objective": "pointwise_bce_plus_within_scene_pairwise_logistic",
              "fitted_splits": [train["split"], calibration["split"]],
              "best_epoch": best[1], "baseline_calibration": baseline,
              "train": metrics(members, train, threshold),
              "calibration": metrics(members, calibration, threshold),
              "calibration_without_context": metrics(members, calibration, threshold,
                                                       use_context=False),
              "calibration_by_slice": sliced_metrics(members, calibration, threshold)}
    saved.update({"members": members.cpu().state_dict(), "decision_threshold": threshold,
                  "paired_ranker": report})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.out)
    print(json.dumps(report, indent=2)); print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
