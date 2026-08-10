"""Export a trained checkpoint's metrics, plus the group decomposition, to results/.

The checkpoint already carries the split metrics `train_latent_dynamics` measured.
What it does not carry is the question those metrics exist to answer: on which
decisions does the imagined rollout beat the compiled plan? That needs the §3
grouping, which depends on the recorded scene rather than on the model:

    A  the plan aims at the landing zone and the grasp held      -> approve
    B  the plan aims off the landing zone                        -> the plan alone decides
    C  the plan aims at the landing zone and the grasp missed    -> only vision decides

Group B is decidable by arithmetic on the action chunk, so a verifier that scores
well overall while failing group C has learned nothing the planner did not know.

    uv run python -m waddle_wm.report_metrics --data data/ur5e_wm_wide \
        --checkpoint models/latent_dynamics_wide.pt --out results/latent_dynamics_wide.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from waddle_wm import windows
from waddle_wm.actions import ACTION_DIM
from waddle_wm.sim.env import LIFT_THRESHOLD, TARGET_RADIUS
from waddle_wm.train_latent_dynamics import Dynamics, Readout, rollout, success_probability

GROUPS = {"A": "plan on the landing zone, grasp holds",
          "B": "plan off the landing zone (the plan decides)",
          "C": "plan on the landing zone, grasp misses (vision decides)"}


def group_of(record) -> str:
    """Which faculty the verdict on this episode needs. See GROUPS."""
    params, before = record["skill"]["params"], record["state_before"]
    on_zone = np.linalg.norm(np.array(params["target_xy"]) - np.array(before["target_pos"])) <= TARGET_RADIUS
    if not on_zone:
        return "B"
    return "A" if max(record["tracks"]["max_block_z"]) > LIFT_THRESHOLD else "C"


def load_model(checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    members = nn.ModuleList([Dynamics(saved["latent_dim"], saved["chunk_dim"]) for _ in range(saved["member_count"])])
    members.load_state_dict(saved["members"])
    readout = Readout(saved["latent_dim"])
    readout.load_state_dict(saved["readout"])
    return saved, members.to(device).eval(), readout.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_wm_wide"))
    ap.add_argument("--checkpoint", type=Path, default=Path("models/latent_dynamics_wide.pt"))
    ap.add_argument("--embeddings", type=Path, default=None, help="default: <data>/window_embeddings.pt")
    ap.add_argument("--out", type=Path, default=Path("results/latent_dynamics_wide.json"))
    ap.add_argument("--chunks", choices=("compiled_plan", "executed"), default="compiled_plan",
                    help="compiled_plan is the verifier's real operating condition")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    manifest = json.loads((args.data / "manifest.json").read_text())
    records = [json.loads(line) for line in (args.data / "records.jsonl").open()]
    cache = torch.load(args.embeddings or args.data / "window_embeddings.pt", weights_only=False)
    saved, members, readout = load_model(args.checkpoint, device)
    if saved["manifest"].get("episodes") != manifest.get("episodes"):
        raise ValueError(f"checkpoint was trained on {saved['manifest'].get('episodes')} episodes, "
                         f"{args.data} has {manifest.get('episodes')}")

    data = windows.build(records, manifest, planned=args.chunks == "compiled_plan")
    norm = saved["normalization"]
    latents = (torch.stack([cache[episode] for episode in data["episode_ids"]]).float() - norm["latent_mean"]) / norm["latent_std"]
    action = (torch.from_numpy(data["action"]) - norm["action_mean"]) / norm["action_std"]
    success = torch.from_numpy(data["success"]).to(device)
    state = torch.from_numpy(data["state"]).to(device)
    latents, action = latents.to(device), action.to(device)
    groups = np.array([group_of(record) for record in records])

    def lifted_by_window(index, trace):
        """Where the grasp bit survives the rollout: imagined vs real latent, per window.

        The oracle column is the same readout on the *real* latent, so a gap between
        them is the dynamics losing information the frozen encoder kept.
        """
        rows = []
        for k, step in enumerate(trace):
            truth = state[index, k + 1, 6]
            imagined = (readout(step)[1][..., 0].sigmoid().mean(0) >= 0.5).float()
            real = (readout(latents[index, k + 1])[1][..., 0].sigmoid() >= 0.5).float()
            rows.append({"window": k + 1,
                         "imagined_lifted_accuracy": float((imagined == truth).float().mean()),
                         "oracle_lifted_accuracy": float((real == truth).float().mean()),
                         "base_rate": float(max(truth.mean(), 1 - truth.mean()))})
        return rows

    report, per_window = {}, {}
    for split in ("train", "val", "test"):
        mask = torch.from_numpy(data["splits"] == split).to(device)
        index = mask.nonzero().flatten()
        with torch.inference_mode():
            final, trace = rollout(members, latents[index, 0], [action[index, k] for k in range(data["steps"])])
            scores = success_probability(readout, final)
            per_window[split] = lifted_by_window(index, trace)
        probability, uncertainty = scores.mean(0), scores.std(0)
        correct = (probability >= 0.5).float() == success[index]
        split_groups = groups[(data["splits"] == split)]
        report[split] = {}
        for name in sorted(GROUPS):
            in_group = torch.from_numpy(split_groups == name).to(device)
            if not in_group.any():
                continue
            report[split][name] = {
                "description": GROUPS[name],
                "episodes": int(in_group.sum()),
                "correct": int(correct[in_group].sum()),
                "accuracy": float(correct[in_group].float().mean()),
                "mean_success_probability": float(probability[in_group].mean()),
                "mean_uncertainty": float(uncertainty[in_group].mean()),
                "true_success_rate": float(success[index][in_group].mean()),
            }

    output = {
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "episodes": manifest.get("episodes"),
        "block_spawn_low": manifest.get("block_spawn_low"),
        "block_spawn_high": manifest.get("block_spawn_high"),
        "chunks": args.chunks,
        "group_decomposition": report,
        "lifted_by_window": per_window,
        "checkpoint_metrics": saved["metrics"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"group_decomposition": report, "lifted_by_window": {"test": per_window["test"]}}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
