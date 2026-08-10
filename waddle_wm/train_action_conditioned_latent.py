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
LIFT_THRESHOLD = 0.09
CACHE_NAME = "vjepa2_context_future_embeddings.pt"
# (context gate, action gate): each learned variant reuses one code path.
VARIANTS = {"context_plus_action": (1.0, 1.0), "context_only_no_action": (1.0, 0.0), "action_only": (0.0, 1.0)}


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
    """Split one clip into a context half and a future half, never overlapping."""
    clip = frames(path)
    mid = max(1, len(clip) // 2)
    return sample_window(clip, 0, mid, count), sample_window(clip, mid, len(clip), count)


def dataset_audit(records):
    """Record the label identities that decide how the held-out numbers may be read."""
    rows = [{
        "split": r["split"],
        "success": bool(r["outcome"]["success"]),
        "target_miss": r["outcome"]["failure_mode"] == "target_miss",
        "lifted": r["state_after"]["max_block_z"] > LIFT_THRESHOLD,
        "plan_offset_m": float(np.linalg.norm(np.subtract(r["skill"]["params"]["target_xy"], r["state_after"]["target_pos"]))),
    } for r in records]
    splits = {split: {
        "episodes": sum(row["split"] == split for row in rows),
        "success": sum(row["split"] == split and row["success"] for row in rows),
        "target_miss": sum(row["split"] == split and row["target_miss"] for row in rows),
    } for split in sorted({row["split"] for row in rows})}
    return {
        "episodes": len(rows),
        "splits": splits,
        "identities": {
            "target_miss_equals_not_success": all(row["target_miss"] != row["success"] for row in rows),
            "success_equals_plan_within_target_radius": all(row["success"] == (row["plan_offset_m"] <= TARGET_RADIUS) for row in rows),
            "every_episode_lifted": all(row["lifted"] for row in rows),
            "distinct_target_sites": len({tuple(r["state_after"]["target_pos"]) for r in records}),
        },
    }


class LatentPredictor(nn.Module):
    """Context latent + skill params -> future latent.

    The plan is widened by an encoder before the concat and the context is dropped out.
    Concatenating the raw 2-d plan onto a standardized 1024-d context makes the action
    roughly 23x smaller in norm, and the net then memorizes the context within a handful
    of full-batch steps instead of learning to use the action at all.
    """

    def __init__(self, dim: int, hidden: int = 1024, plan_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.plan_encoder = nn.Linear(2, plan_dim)
        self.context_dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(
            nn.Linear(dim + plan_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, context_embedding, plan):
        return self.net(torch.cat((self.context_dropout(context_embedding), self.plan_encoder(plan)), dim=-1))


class OutcomeHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Linear(dim, len(HEADS) + 2)

    def forward(self, predicted_future):
        return self.net(predicted_future)


def encode_windows(records, args, device):
    """Embed the context and future halves with the frozen trunk, reusing a disk cache."""
    cache_path = args.data / CACHE_NAME
    cached = torch.load(cache_path, weights_only=True) if cache_path.exists() and not args.refresh_cache else {}
    # Embeddings depend on the window frame count, so a changed --frames must not reuse them.
    entries = cached.get("entries", {}) if cached.get("frames") == args.frames else {}
    missing = [r for r in records if r["episode_id"] not in entries]
    if missing:
        processor = AutoVideoProcessor.from_pretrained(args.model, local_files_only=True)
        encoder = AutoModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
        with torch.inference_mode():
            for i, record in enumerate(missing, 1):
                context, future = load_pair(args.data / record["observation"]["frames_path"], args.frames)
                context_inputs = {k: v.to(device) for k, v in processor(context, return_tensors="pt").items()}
                future_inputs = {k: v.to(device) for k, v in processor(future, return_tensors="pt").items()}
                entries[record["episode_id"]] = {
                    "context": encoder(**context_inputs).last_hidden_state.mean(dim=1).squeeze(0).float().cpu(),
                    "future": encoder(**future_inputs).last_hidden_state.mean(dim=1).squeeze(0).float().cpu(),
                }
                print(f"embedded {i}/{len(missing)}", flush=True)
        torch.save({"frames": args.frames, "entries": entries}, cache_path)
    context = torch.stack([entries[r["episode_id"]]["context"] for r in records]).float()
    future = torch.stack([entries[r["episode_id"]]["future"] for r in records]).float()
    return context, future


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_tabletop"))
    ap.add_argument("--model", type=Path, default=Path("models/vjepa2-vitl-fpc64-256"))
    ap.add_argument("--out", type=Path, default=Path("models/action_conditioned_latent.pt"))
    ap.add_argument("--report", type=Path, default=Path("results/action_conditioned_latent.json"))
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--plan-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    records = [json.loads(line) for line in (args.data / "records.jsonl").open()]
    if args.max_episodes:
        records = records[:args.max_episodes]
    audit = dataset_audit(records)
    print(json.dumps({"dataset_audit": audit}, indent=2), flush=True)

    context, future = encode_windows(records, args, device)
    plan = torch.tensor([r["skill"]["params"]["target_xy"] for r in records]).float()
    labels = torch.tensor([[r["outcome"]["success"], r["outcome"]["failure_mode"] == "target_miss"] for r in records]).float()
    pos = torch.tensor([r["state_after"]["block_pos"][:2] for r in records]).float()
    pos_before = torch.tensor([r["state_before"]["block_pos"][:2] for r in records]).float()
    splits = [r["split"] for r in records]
    train = torch.tensor([s == "train" for s in splits]); val = torch.tensor([s == "val" for s in splits]); test = torch.tensor([s == "test" for s in splits])
    if not (train.any() and val.any() and test.any()):
        raise ValueError("records need non-empty train, val, and test splits")

    ctx_mean, ctx_std = context[train].mean(0), context[train].std(0).clamp_min(1e-6)
    fut_mean, fut_std = future[train].mean(0), future[train].std(0).clamp_min(1e-6)
    plan_mean, plan_std = plan[train].mean(0), plan[train].std(0).clamp_min(1e-6)
    pos_mean, pos_std = pos[train].mean(0), pos[train].std(0).clamp_min(1e-6)
    context_raw = context
    context = (context - ctx_mean) / ctx_std
    future = (future - fut_mean) / fut_std
    plan = (plan - plan_mean) / plan_std
    pos = (pos - pos_mean) / pos_std

    tensors = (context, context_raw, future, plan, labels, pos, pos_before)
    context, context_raw, future, plan, labels, pos, pos_before = (t.to(device) for t in tensors)
    fut_mean_d, fut_std_d = fut_mean.to(device), fut_std.to(device)
    pos_mean_d, pos_std_d = pos_mean.to(device), pos_std.to(device)

    def score(mask, predicted_future=None, label_probs=None, predicted_pos=None):
        """Held-out metrics for any predictor, learned or analytic. Positions are raw metres."""
        result = {"episodes": int(mask.sum()), "success_rate": float(labels[mask, 0].mean()), "target_radius_m": TARGET_RADIUS}
        if predicted_future is not None:
            # Scored on train-standardized latents: raw V-JEPA latents share a large common
            # component, so raw cosine reads ~0.99 for every predictor including the mean.
            result["future_latent_mse"] = float(nn.functional.mse_loss(predicted_future, future[mask]))
            result["future_latent_cosine_centered"] = float(nn.functional.cosine_similarity(predicted_future, future[mask], dim=-1).mean())
        if label_probs is not None:
            for i, name in enumerate(HEADS):
                result[f"{name}_accuracy"] = float(((label_probs[:, i] >= .5) == labels[mask, i].bool()).float().mean())
        if predicted_pos is not None:
            actual_pos = pos[mask] * pos_std_d + pos_mean_d
            result["block_xy_rmse_m"] = float(torch.sqrt(nn.functional.mse_loss(predicted_pos, actual_pos)))
            result["block_xy_mean_error_m"] = float((predicted_pos - actual_pos).norm(dim=-1).mean())
        return result

    def fit(use_context: float, use_action: float):
        """Train one predictor+head variant with the context and/or action input gated off."""
        torch.manual_seed(args.seed)
        predictor = LatentPredictor(context.shape[1], plan_dim=args.plan_dim, dropout=args.dropout).to(device)
        head = OutcomeHead(future.shape[1]).to(device)
        opt = torch.optim.AdamW([*predictor.parameters(), *head.parameters()], lr=1e-3, weight_decay=1e-4)

        def forward(mask):
            predicted_future = predictor(context[mask] * use_context, plan[mask] * use_action)
            return predicted_future, head(predicted_future)

        def objective(mask):
            predicted_future, decoded = forward(mask)
            latent_loss = nn.functional.mse_loss(predicted_future, future[mask])
            label_loss = nn.functional.binary_cross_entropy_with_logits(decoded[:, :len(HEADS)], labels[mask])
            pos_loss = nn.functional.mse_loss(decoded[:, len(HEADS):], pos[mask])
            return latent_loss + label_loss + pos_loss

        best = (float("inf"), 0, {k: v.clone() for k, v in predictor.state_dict().items()}, {k: v.clone() for k, v in head.state_dict().items()})
        for epoch in range(1, args.epochs + 1):
            predictor.train()
            opt.zero_grad(); objective(train).backward(); opt.step()
            predictor.eval()  # dropout must be off for model selection and for every reported metric
            with torch.inference_mode():
                val_loss = float(objective(val))
            if val_loss < best[0]:
                best = (val_loss, epoch, {k: v.clone() for k, v in predictor.state_dict().items()}, {k: v.clone() for k, v in head.state_dict().items()})
        predictor.load_state_dict(best[2]); head.load_state_dict(best[3])
        predictor.eval()

        def evaluate(mask):
            with torch.inference_mode():
                predicted_future, decoded = forward(mask)
                return score(mask, predicted_future, decoded[:, :len(HEADS)].sigmoid(), decoded[:, len(HEADS):] * pos_std_d + pos_mean_d)

        report = {"best_epoch": best[1], "best_val_loss": best[0]}
        report.update({split_name: evaluate(mask) for split_name, mask in (("train", train), ("val", val), ("test", test))})
        return predictor, head, report

    def constant_baseline(mask):
        """Train-set majority label, train-mean future latent, train-mean block position."""
        rate = labels[train].mean(0)
        return score(
            mask,
            torch.zeros(int(mask.sum()), future.shape[1], device=device),  # train mean is 0 in normalized space
            rate.expand(int(mask.sum()), -1),
            (pos[train].mean(0) * pos_std_d + pos_mean_d).expand(int(mask.sum()), -1),
        )

    def persistence_baseline(mask):
        """Nothing moves: future latent = context latent, final block xy = block xy before the skill."""
        return score(
            mask,
            (context_raw[mask] - fut_mean_d) / fut_std_d,
            labels[train].mean(0).round().expand(int(mask.sum()), -1),  # no persistence notion for labels; use majority
            pos_before[mask],
        )

    models = {}
    for name, (use_context, use_action) in VARIANTS.items():
        predictor, head, report = fit(use_context, use_action)
        models[name] = report
        print(json.dumps({name: report}, indent=2), flush=True)
        if name == "context_plus_action":
            best_predictor, best_head = predictor, head

    result = {
        "episodes": len(records),
        "frames_per_window": args.frames,
        "seed": args.seed,
        "config": {"epochs": args.epochs, "plan_dim": args.plan_dim, "dropout": args.dropout, "lr": 1e-3, "weight_decay": 1e-4},
        "dataset_audit": audit,
        "models": models,
        "baselines": {
            "constant_majority_mean": {name: constant_baseline(mask) for name, mask in (("train", train), ("val", val), ("test", test))},
            "persistence": {name: persistence_baseline(mask) for name, mask in (("train", train), ("val", val), ("test", test))},
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "predictor_state_dict": best_predictor.cpu().state_dict(),
        "head_state_dict": best_head.cpu().state_dict(),
        "embedding_dim": context.shape[1],
        "heads": list(HEADS),
        "frames": args.frames,
        "plan_dim": args.plan_dim,
        "dropout": args.dropout,
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
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
