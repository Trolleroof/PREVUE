"""Train a frozen-V-JEPA latent predictor from context frames plus skill params."""
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

HEADS = ("success", "target_miss")


def frames(path: Path):
    cap, clip = cv2.VideoCapture(str(path)), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        clip.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not clip:
        raise ValueError(f"no frames in {path}")
    return clip


def sample_window(clip, start, stop, count):
    window = clip[start:stop] or [clip[min(start, len(clip) - 1)]]
    idx = np.linspace(0, len(window) - 1, count).round().astype(int)
    return [window[i] for i in idx]


def load_pair(path: Path, count: int):
    clip = frames(path)
    mid = max(1, len(clip) // 2)
    return sample_window(clip, 0, mid, count), sample_window(clip, mid, len(clip), count)


class LatentPredictor(nn.Module):
    def __init__(self, dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, context_embedding, plan):
        return self.net(torch.cat((context_embedding, plan), dim=-1))


class OutcomeHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Linear(dim, len(HEADS) + 2)

    def forward(self, predicted_future):
        return self.net(predicted_future)


def encode_windows(records, args, device):
    cache_path = args.data / "vjepa2_context_future_embeddings.pt"
    cache = torch.load(cache_path) if cache_path.exists() and not args.refresh_cache else {}
    missing = [r for r in records if r["episode_id"] not in cache]
    if missing:
        processor = AutoVideoProcessor.from_pretrained(args.model, local_files_only=True)
        encoder = AutoModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
        with torch.inference_mode():
            for i, record in enumerate(missing, 1):
                context, future = load_pair(args.data / record["observation"]["frames_path"], args.frames)
                context_inputs = {k: v.to(device) for k, v in processor(context, return_tensors="pt").items()}
                future_inputs = {k: v.to(device) for k, v in processor(future, return_tensors="pt").items()}
                cache[record["episode_id"]] = {
                    "context": encoder(**context_inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu(),
                    "future": encoder(**future_inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu(),
                }
                print(f"embedded {i}/{len(missing)}", flush=True)
        torch.save(cache, cache_path)
    context = torch.stack([cache[r["episode_id"]]["context"] for r in records]).float()
    future = torch.stack([cache[r["episode_id"]]["future"] for r in records]).float()
    return context, future


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_tabletop"))
    ap.add_argument("--model", type=Path, default=Path("models/vjepa2-vitl-fpc64-256"))
    ap.add_argument("--out", type=Path, default=Path("models/action_conditioned_latent.pt"))
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    records = [json.loads(line) for line in (args.data / "records.jsonl").open()]
    if args.max_episodes:
        records = records[:args.max_episodes]
    context, future = encode_windows(records, args, device)
    plan = torch.tensor([r["skill"]["params"]["target_xy"] for r in records]).float()
    labels = torch.tensor([[r["outcome"]["success"], r["outcome"]["failure_mode"] == "target_miss"] for r in records]).float()
    pos = torch.tensor([r["state_after"]["block_pos"][:2] for r in records]).float()
    splits = [r["split"] for r in records]
    train = torch.tensor([s == "train" for s in splits]); val = torch.tensor([s == "val" for s in splits]); test = torch.tensor([s == "test" for s in splits])
    if not (train.any() and val.any() and test.any()):
        raise ValueError("records need non-empty train, val, and test splits")

    ctx_mean, ctx_std = context[train].mean(0), context[train].std(0).clamp_min(1e-6)
    fut_mean, fut_std = future[train].mean(0), future[train].std(0).clamp_min(1e-6)
    plan_mean, plan_std = plan[train].mean(0), plan[train].std(0).clamp_min(1e-6)
    pos_mean, pos_std = pos[train].mean(0), pos[train].std(0).clamp_min(1e-6)
    context = (context - ctx_mean) / ctx_std
    future = (future - fut_mean) / fut_std
    plan = (plan - plan_mean) / plan_std
    pos = (pos - pos_mean) / pos_std

    predictor = LatentPredictor(context.shape[1]).to(device)
    head = OutcomeHead(future.shape[1]).to(device)
    opt = torch.optim.AdamW([*predictor.parameters(), *head.parameters()], lr=1e-3, weight_decay=1e-4)
    context, future, plan, labels, pos = (t.to(device) for t in (context, future, plan, labels, pos))
    pos_mean_d, pos_std_d = pos_mean.to(device), pos_std.to(device)

    def objective(mask):
        predicted_future = predictor(context[mask], plan[mask])
        decoded = head(predicted_future)
        latent_loss = nn.functional.mse_loss(predicted_future, future[mask])
        label_loss = nn.functional.binary_cross_entropy_with_logits(decoded[:, :len(HEADS)], labels[mask])
        pos_loss = nn.functional.mse_loss(decoded[:, len(HEADS):], pos[mask])
        return latent_loss + label_loss + pos_loss

    def metrics(mask):
        with torch.inference_mode():
            predicted_future = predictor(context[mask], plan[mask])
            decoded = head(predicted_future)
            probs = decoded[:, :len(HEADS)].sigmoid()
            predicted_pos = decoded[:, len(HEADS):] * pos_std_d + pos_mean_d
            actual_pos = pos[mask] * pos_std_d + pos_mean_d
            result = {f"{name}_accuracy": float(((probs[:, i] >= .5) == labels[mask, i].bool()).float().mean()) for i, name in enumerate(HEADS)}
            result["future_latent_mse"] = float(nn.functional.mse_loss(predicted_future, future[mask]))
            result["block_xy_rmse_m"] = float(torch.sqrt(nn.functional.mse_loss(predicted_pos, actual_pos)))
            result["success_rate"] = float(labels[mask, 0].mean())
            result["target_radius_m"] = TARGET_RADIUS
            return result

    best = (float("inf"), 0, predictor.state_dict(), head.state_dict())
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(); objective(train).backward(); opt.step()
        with torch.inference_mode():
            val_loss = float(objective(val))
        if val_loss < best[0]:
            best = (
                val_loss,
                epoch,
                {k: v.clone() for k, v in predictor.state_dict().items()},
                {k: v.clone() for k, v in head.state_dict().items()},
            )
    predictor.load_state_dict(best[2]); head.load_state_dict(best[3])
    result = {"train": metrics(train), "val": metrics(val), "test": metrics(test), "episodes": len(records), "best_epoch": best[1], "best_val_loss": best[0]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "predictor_state_dict": predictor.cpu().state_dict(),
        "head_state_dict": head.cpu().state_dict(),
        "embedding_dim": context.shape[1],
        "heads": list(HEADS),
        "frames": args.frames,
        "normalization": {
            "context_mean": ctx_mean,
            "context_std": ctx_std,
            "future_mean": fut_mean,
            "future_std": fut_std,
            "plan_mean": plan_mean,
            "plan_std": plan_std,
            "position_mean": pos_mean,
            "position_std": pos_std,
        },
    }, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
