"""Train a frozen-V-JEPA outcome probe for UR5e pick-and-place records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoVideoProcessor

from waddle_wm.sim.env import TARGET_RADIUS

HEADS = ("success", "target_miss", "in_target")
LIFT_THRESHOLD = 0.09


def frames(path: Path, count: int = 32):
    """Sample `count` frames spread evenly over the whole clip."""
    cap, clip = cv2.VideoCapture(str(path)), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        clip.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not clip:
        raise ValueError(f"no frames in {path}")
    return [clip[i] for i in np.linspace(0, len(clip) - 1, count).round().astype(int)]


def label_audit(records):
    """Summarize label coverage and identities before training."""
    labels = []
    for record in records:
        outcome, after = record["outcome"], record["state_after"]
        lifted = after["max_block_z"] > LIFT_THRESHOLD
        in_target = after["target_distance"] <= TARGET_RADIUS
        labels.append({"split": record["split"], "success": bool(outcome["success"]),
                       "lifted": lifted, "in_target": in_target,
                       "failure_mode": outcome["failure_mode"]})

    def counts(rows, key):
        return {str(value): sum(row[key] == value for row in rows) for value in sorted({row[key] for row in rows}, key=str)}

    def slices(rows):
        return {
            "lifted_and_in_target": sum(row["lifted"] and row["in_target"] for row in rows),
            "lifted_and_missed_target": sum(row["lifted"] and not row["in_target"] for row in rows),
            "failed_grasp_or_dropped_block": sum(not row["lifted"] for row in rows),
            "target_miss": sum(row["failure_mode"] == "target_miss" for row in rows),
            "grasp_failure": sum(not row["lifted"] and row["failure_mode"] != "target_miss" for row in rows),
        }

    result = {"episodes": len(labels), "splits": {}, "failure_modes": counts(labels, "failure_mode"),
              "label_balance": {key: counts(labels, key) for key in ("success", "lifted", "in_target")},
              "evaluation_slices": slices(labels)}
    for split in sorted({row["split"] for row in labels}):
        rows = [row for row in labels if row["split"] == split]
        result["splits"][split] = {"episodes": len(rows), "failure_modes": counts(rows, "failure_mode"),
                                   "label_balance": {key: counts(rows, key) for key in ("success", "lifted", "in_target")},
                                   "evaluation_slices": slices(rows)}
    result["identities"] = {
        "success_equals_in_target": all(row["success"] == row["in_target"] for row in labels),
        "success_equals_lifted_and_in_target": all(row["success"] == (row["lifted"] and row["in_target"]) for row in labels),
    }
    result["in_target_status"] = "provisional: identical to success on this dataset" if result["identities"]["success_equals_in_target"] else "independent variation present"
    return result


class Probe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Linear(dim + 2, len(HEADS) + 2)  # success, target_miss, in_target, future block xy

    def forward(self, embedding, plan):
        return self.head(torch.cat((embedding, plan), dim=-1))


def embed(records, args, device):
    """Encode every clip with the frozen V-JEPA trunk, reusing a disk cache."""
    cache_path = args.data / "vjepa2_embeddings.pt"
    cache = torch.load(cache_path) if cache_path.exists() and not args.refresh_cache else {}
    missing = [r for r in records if r["episode_id"] not in cache]
    if missing:
        processor = AutoVideoProcessor.from_pretrained(args.model, local_files_only=True)
        encoder = AutoModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
        with torch.inference_mode():
            for i, record in enumerate(missing, 1):
                clip = frames(args.data / record["observation"]["frames_path"], args.frames)
                inputs = {k: v.to(device) for k, v in processor(clip, return_tensors="pt").items()}
                cache[record["episode_id"]] = encoder(**inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu()
                print(f"embedded {i}/{len(missing)}", flush=True)
        torch.save(cache, cache_path)
    return torch.stack([cache[r["episode_id"]] for r in records]).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_tabletop"))
    ap.add_argument("--model", type=Path, default=Path("models/vjepa2-vitl-fpc64-256"))
    ap.add_argument("--out", type=Path, default=Path("models/outcome_probe.pt"))
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--refresh-cache", action="store_true", help="re-encode clips instead of reusing the cache")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    records = [json.loads(line) for line in (args.data / "records.jsonl").open()]
    if args.max_episodes:
        records = records[:args.max_episodes]
    audit = label_audit(records)
    print(json.dumps({"label_audit": audit}, indent=2))
    x = embed(records, args, device)
    plan = torch.tensor([r["skill"]["params"]["target_xy"] for r in records]).float()
    y = torch.tensor([[r["outcome"]["success"], r["outcome"]["failure_mode"] == "target_miss",
                       r["state_after"]["target_distance"] <= TARGET_RADIUS] for r in records]).float()
    pos = torch.tensor([r["state_after"]["block_pos"][:2] for r in records]).float()
    splits = [r["split"] for r in records]
    train = torch.tensor([s == "train" for s in splits]); val = torch.tensor([s == "val" for s in splits]); test = torch.tensor([s == "test" for s in splits])
    if not (train.any() and val.any() and test.any()):
        raise ValueError("records need non-empty train, val, and test splits")
    x_mean, x_std = x[train].mean(0), x[train].std(0).clamp_min(1e-6)
    plan_mean, plan_std = plan[train].mean(0), plan[train].std(0).clamp_min(1e-6)
    pos_mean, pos_std = pos[train].mean(0), pos[train].std(0).clamp_min(1e-6)
    x, plan, pos = (x - x_mean) / x_std, (plan - plan_mean) / plan_std, (pos - pos_mean) / pos_std
    probe = Probe(x.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    x, plan, y, pos, pos_mean_d, pos_std_d = (t.to(device) for t in (x, plan, y, pos, pos_mean, pos_std))

    def objective(mask):
        out = probe(x[mask], plan[mask])
        return nn.functional.binary_cross_entropy_with_logits(out[:, :len(HEADS)], y[mask]) + nn.functional.mse_loss(out[:, len(HEADS):], pos[mask])

    def metrics(mask):
        with torch.inference_mode():
            out = probe(x[mask], plan[mask]); probs = out[:, :len(HEADS)].sigmoid()
            predicted_pos = out[:, len(HEADS):] * pos_std_d + pos_mean_d
            actual_pos = pos[mask] * pos_std_d + pos_mean_d
            result = {f"{name}_accuracy": float(((probs[:, i] >= .5) == y[mask, i].bool()).float().mean()) for i, name in enumerate(HEADS)}
            result["in_target_status"] = audit["in_target_status"]
            result["block_xy_rmse_m"] = float(torch.sqrt(nn.functional.mse_loss(predicted_pos, actual_pos)))
            return result

    best = (float("inf"), 0, {k: v.clone() for k, v in probe.state_dict().items()})
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(); objective(train).backward(); opt.step()
        with torch.inference_mode():
            val_loss = float(objective(val))
        if val_loss < best[0]:
            best = (val_loss, epoch, {k: v.clone() for k, v in probe.state_dict().items()})
    probe.load_state_dict(best[2])
    result = {"label_audit": audit, "train": metrics(train), "val": metrics(val), "test": metrics(test),
              "episodes": len(records), "best_epoch": best[1], "best_val_loss": best[0]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": probe.cpu().state_dict(), "embedding_dim": x.shape[1], "heads": list(HEADS), "provisional_heads": ["in_target"], "frames": args.frames,
                "normalization": {"embedding_mean": x_mean, "embedding_std": x_std, "plan_mean": plan_mean, "plan_std": plan_std,
                                  "position_mean": pos_mean, "position_std": pos_std}}, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
