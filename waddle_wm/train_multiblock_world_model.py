"""Train the multi-block state world model used by the live verifier.

Current camera state + frozen V-JEPA context + compiled skill trace -> future XYZ
for every block plus grasp and task-success probabilities.

The plan half of that input is `waddle_wm.plan_encoding`, and the checkpoint records which
version it was fitted with plus whether the corpus ever commanded a wrist heading. A corpus
that never did produces a checkpoint the readers refuse as orientation-blind, and the trainer
says so at the end of the run rather than leaving it to be discovered by a flat ranking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from waddle_wm import plan_encoding, windows
from waddle_wm.sim import relling_scene as scene
from waddle_wm.train_latent_dynamics import plan_only_baseline, rules_baseline


class StateWorldModel(nn.Module):
    def __init__(self, context_dim: int, plan_dim: int, task_dim: int = 7, hidden: int = 256):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(context_dim, 32), nn.GELU())
        self.plan = nn.Sequential(nn.Linear(plan_dim, 32), nn.GELU())
        self.net = nn.Sequential(nn.Linear(32 + 32 + 9 + task_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.state = nn.Linear(hidden, 9)
        self.outcome = nn.Linear(hidden, 2)

    def forward(self, context, state, plan, task):
        hidden = self.net(torch.cat([self.context(context), self.plan(plan), state, task], dim=-1))
        return state + self.state(hidden), self.outcome(hidden)


def task_features(objects, destinations, block_names, device):
    destination_names = ("green_pad", *block_names)
    source = torch.tensor([block_names.index(name) for name in objects], device=device)
    destination = torch.tensor([destination_names.index(name) for name in destinations], device=device)
    return torch.cat([nn.functional.one_hot(source, len(block_names)),
                      nn.functional.one_hot(destination, len(destination_names))], dim=-1).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_wm_multiblock"))
    ap.add_argument("--embeddings", type=Path)
    ap.add_argument("--out", type=Path, default=Path("models/multiblock_world_model.pt"))
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outcome-weight", type=float, default=1.0)
    ap.add_argument("--plan-encoding", type=int, default=plan_encoding.PLAN_ENCODING_VERSION,
                    choices=sorted(plan_encoding.PLAN_FIELDS),
                    help="plan vector version; 1 is the orientation-blind encoding kept for replay")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    manifest = json.loads((args.data / "manifest.json").read_text())
    records = [json.loads(line) for line in (args.data / "records.jsonl").open()]
    cache = torch.load(args.embeddings or args.data / "window_embeddings.pt", weights_only=False)
    data = windows.build(records, manifest, planned=True)
    block_names = tuple(manifest["block_names"])

    context = torch.stack([cache[episode][0] for episode in data["episode_ids"]]).float()
    all_state = torch.from_numpy(data["state"])[..., :9]
    initial, target = all_state[:, 0], all_state[:, -1]
    plan_rows = []
    for episode, record in enumerate(records):
        source = block_names.index(data["objects"][episode])
        source_xyz = initial[episode, source * 3:source * 3 + 3]
        destination = data["destinations"][episode]
        destination_xyz = (torch.tensor(scene.TARGET_POS) if destination == "green_pad" else
                           initial[episode, block_names.index(destination) * 3:block_names.index(destination) * 3 + 3])
        trace = record["skill"]["trace"]
        grasp = next(entry["target"] for entry in trace if entry["phase"] == "descend")
        place = next(entry["target"] for entry in trace if entry["phase"] == "place")
        grasp_yaw, approach_yaw = plan_encoding.trace_yaws(trace)
        plan_rows.append(torch.from_numpy(plan_encoding.plan_vector(
            grasp, place, source_xyz.numpy(), destination_xyz.numpy(),
            grasp_yaw, approach_yaw, args.plan_encoding)))
    plan = torch.stack(plan_rows).float()
    encoding = plan_encoding.yaw_informative(plan.numpy(), data["splits"] == "train",
                                             args.plan_encoding)
    success = torch.from_numpy(data["success"])
    grasped = torch.tensor([max(record["tracks"]["max_block_z"]) > 0.09 for record in records], dtype=torch.float32)
    labels = torch.stack([grasped, success], dim=-1)
    split = {name: torch.from_numpy(data["splits"] == name) for name in ("train", "val", "test")}
    train = split["train"]

    def norm(tensor):
        mean, std = tensor[train].mean(0), tensor[train].std(0).clamp_min(1e-6)
        return (tensor - mean) / std, mean, std

    context, context_mean, context_std = norm(context)
    plan, plan_mean, plan_std = norm(plan)
    state_train = torch.cat([initial[train], target[train]])
    state_mean, state_std = state_train.mean(0), state_train.std(0).clamp_min(1e-6)
    initial = (initial - state_mean) / state_std
    target = (target - state_mean) / state_std
    task = task_features(data["objects"], data["destinations"], block_names, device)
    context, plan, initial, target, success, labels = (x.to(device) for x in
                                                       (context, plan, initial, target, success, labels))
    split = {name: mask.to(device) for name, mask in split.items()}

    members = nn.ModuleList([StateWorldModel(context.shape[1], plan.shape[1]) for _ in range(args.members)]).to(device)
    optimizer = torch.optim.AdamW(members.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    train_index = split["train"].nonzero().flatten()
    bootstrap = [train_index[torch.randint(len(train_index), (len(train_index),), generator=generator)].to(device)
                 for _ in members]

    source_ids = torch.tensor([block_names.index(name) for name in data["objects"]], device=device)

    def loss(indices):
        total = 0.0
        for member, index in zip(members, indices):
            predicted, logits = member(context[index], initial[index], plan[index], task[index])
            selected = source_ids[index, None] * 3 + torch.arange(3, device=device)
            total += nn.functional.mse_loss(predicted, target[index]) + nn.functional.mse_loss(
                predicted.gather(1, selected), target[index].gather(1, selected)) + args.outcome_weight * nn.functional.binary_cross_entropy_with_logits(
                logits, labels[index])
        return total / len(members)

    val_index = split["val"].nonzero().flatten()
    best = (float("inf"), 0, None)
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(); loss(bootstrap).backward(); optimizer.step()
        with torch.inference_mode():
            value = float(loss([val_index] * len(members)))
        if value < best[0]:
            best = (value, epoch, {key: value.clone() for key, value in members.state_dict().items()})
        if epoch % 200 == 0:
            print(f"epoch {epoch}: val {value:.4f} (best {best[0]:.4f} @ {best[1]})", flush=True)
    members.load_state_dict(best[2]); members.eval()

    def scores_for(mask, use_context=True):
        index = mask.nonzero().flatten()
        with torch.inference_mode():
            outputs = [member(context[index] if use_context else torch.zeros_like(context[index]),
                              initial[index], plan[index], task[index]) for member in members]
            predictions = torch.stack([output[0] for output in outputs])
            logits = torch.stack([output[1] for output in outputs])
        return index, predictions, logits

    def evaluate(mask, threshold=0.5, use_context=True):
        index, predictions, logits = scores_for(mask, use_context)
        positions = predictions * state_std.to(device) + state_mean.to(device)
        truth_position = target[index] * state_std.to(device) + state_mean.to(device)
        errors = []
        episodes = index.detach().cpu().tolist()
        for column, episode in enumerate(episodes):
            source = block_names.index(data["objects"][episode])
            source_xyz = positions[:, column, source * 3:source * 3 + 3]
            actual = truth_position[column, source * 3:source * 3 + 2]
            errors.append((source_xyz.mean(0)[:2] - actual).norm())
        errors = torch.stack(errors)
        scores = logits[..., 1].sigmoid()
        probability, uncertainty = scores.mean(0), scores.std(0)
        truth = success[index]
        verdict, correct = (probability >= threshold).float(), None
        correct = verdict == truth
        grasp_probability = logits[..., 0].sigmoid().mean(0)
        return {"episodes": len(index), "success_accuracy": float(correct.float().mean()),
                "grasp_accuracy": float(((grasp_probability >= 0.5).float() == labels[index, 0]).float().mean()),
                "selected_block_xy_rmse_m": float(errors.pow(2).mean().sqrt()),
                "brier": float((probability - truth).pow(2).mean()),
                "false_accept_rate": float(((verdict == 1) & (truth == 0)).float().sum() / max(1, int((truth == 0).sum()))),
                "false_reject_rate": float(((verdict == 0) & (truth == 1)).float().sum() / max(1, int((truth == 1).sum()))),
                "mean_uncertainty": float(uncertainty.mean()), "threshold": threshold}

    val_index, _, val_logits = scores_for(split["val"])
    val_probability = val_logits[..., 1].sigmoid().mean(0)
    val_truth = success[val_index]
    candidates = torch.unique(val_probability).sort().values.tolist()
    safe = [value for value in candidates if float(
        (((val_probability >= value) & (val_truth == 0)).float().sum() /
         max(1, int((val_truth == 0).sum())))) <= 0.10]
    decision_threshold = max(safe, key=lambda value: float(
        ((val_probability >= value).float() == val_truth).float().mean())) if safe else 1.0

    metrics = {"plan_only_baseline": plan_only_baseline(data, data, data["splits"]),
               "rules_baseline": rules_baseline(records), "best_epoch": best[1], "best_val_loss": best[0],
               "outcome_weight": args.outcome_weight,
               "decision_threshold": decision_threshold,
               "plan_encoding": encoding,
               "test": evaluate(split["test"], decision_threshold),
               "test_without_vjepa_context": evaluate(split["test"], decision_threshold, False)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "multiblock_state", "members": members.cpu().state_dict(),
                "context_dim": context.shape[1], "plan_dim": plan.shape[1], "member_count": len(members),
                "manifest": manifest, "normalization": {"context_mean": context_mean, "context_std": context_std,
                    "plan_mean": plan_mean, "plan_std": plan_std, "state_mean": state_mean, "state_std": state_std},
                "plan_encoding": encoding,
                "decision_threshold": decision_threshold, "metrics": metrics}, args.out)
    print(json.dumps(metrics, indent=2))
    if plan_encoding.orientation_blind(plan_encoding.declared({"plan_encoding": encoding})):
        print(f"\nWARNING: this checkpoint is orientation-blind — "
              f"{plan_encoding.blindness_reason(encoding)}. The visual selector and the verifier "
              f"will refuse it for orientation-dependent ranking unless they are asked to accept "
              f"the limitation explicitly. Train on a corpus that commands a grasp yaw "
              f"(`generate_dataset --oriented`) to get a yaw-aware checkpoint.", flush=True)


if __name__ == "__main__":
    main()
