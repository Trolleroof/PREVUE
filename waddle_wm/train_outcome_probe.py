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


def frames(path: Path, count: int = 16):
    cap, result = cv2.VideoCapture(str(path)), []
    while len(result) < count:
        ok, frame = cap.read()
        if not ok:
            break
        result.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not result:
        raise ValueError(f"no frames in {path}")
    return result + [result[-1]] * (count - len(result))


class Probe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Linear(dim + 2, 4)  # success, target_miss, future block xy

    def forward(self, embedding, plan):
        return self.head(torch.cat((embedding, plan), dim=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_tabletop"))
    ap.add_argument("--model", type=Path, default=Path(".hf-cache/hub/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f"))
    ap.add_argument("--out", type=Path, default=Path("models/outcome_probe.pt"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max-episodes", type=int)
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    records = [json.loads(line) for line in (args.data / "records.jsonl").open()]
    if args.max_episodes:
        records = records[:args.max_episodes]
    processor = AutoVideoProcessor.from_pretrained(args.model, local_files_only=True)
    encoder = AutoModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
    embeddings, plans, labels, positions, splits = [], [], [], [], []
    with torch.inference_mode():
        for i, record in enumerate(records, 1):
            clip = frames(args.data / record["observation"]["frames_path"])
            inputs = {k: v.to(device) for k, v in processor(clip, return_tensors="pt").items()}
            embeddings.append(encoder(**inputs).last_hidden_state.mean(dim=1).cpu())
            plans.append(record["skill"]["params"]["target_xy"])
            labels.append([record["outcome"]["success"], record["outcome"]["failure_mode"] == "target_miss"])
            positions.append(record["state_after"]["block_pos"][:2])
            splits.append(record["split"])
            print(f"embedded {i}/{len(records)}", flush=True)
    x = torch.cat(embeddings).float(); plan = torch.tensor(plans).float()
    y = torch.tensor(labels).float(); pos = torch.tensor(positions).float()
    train = torch.tensor([s == "train" for s in splits]); val = torch.tensor([s == "val" for s in splits]); test = torch.tensor([s == "test" for s in splits])
    if not (train.any() and val.any() and test.any()):
        raise ValueError("records need non-empty train, val, and test splits")
    x_mean, x_std = x[train].mean(0), x[train].std(0).clamp_min(1e-6)
    plan_mean, plan_std = plan[train].mean(0), plan[train].std(0).clamp_min(1e-6)
    pos_mean, pos_std = pos[train].mean(0), pos[train].std(0).clamp_min(1e-6)
    x, plan, pos = (x - x_mean) / x_std, (plan - plan_mean) / plan_std, (pos - pos_mean) / pos_std
    probe = Probe(x.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    x, plan, y, pos, pos_mean, pos_std = (t.to(device) for t in (x, plan, y, pos, pos_mean, pos_std))
    for _ in range(args.epochs):
        out = probe(x[train], plan[train])
        loss = nn.functional.binary_cross_entropy_with_logits(out[:, :2], y[train]) + nn.functional.mse_loss(out[:, 2:], pos[train])
        opt.zero_grad(); loss.backward(); opt.step()
    def metrics(mask):
        with torch.inference_mode():
            out = probe(x[mask], plan[mask]); probs = out[:, :2].sigmoid()
            predicted_pos, actual_pos = out[:, 2:] * pos_std + pos_mean, pos[mask] * pos_std + pos_mean
            return {"success_accuracy": float(((probs[:, 0] >= .5) == y[mask, 0].bool()).float().mean()), "target_miss_accuracy": float(((probs[:, 1] >= .5) == y[mask, 1].bool()).float().mean()), "block_xy_rmse_m": float(torch.sqrt(nn.functional.mse_loss(predicted_pos, actual_pos)))}
    result = {"val": metrics(val), "test": metrics(test), "episodes": len(records)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": probe.cpu().state_dict(), "embedding_dim": x.shape[1], "normalization": {"embedding_mean": x_mean, "embedding_std": x_std, "plan_mean": plan_mean, "plan_std": plan_std, "position_mean": pos_mean.cpu(), "position_std": pos_std.cpu()}}, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
