"""Cache frozen V-JEPA 2 latents, one per 8-frame window, for every episode.

The trunk is never trained, so this runs once and everything downstream works on
a (windows, 1024) tensor per episode.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor


def clip_frames(path: Path, expected: int) -> list[np.ndarray]:
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != expected:
        raise ValueError(f"{path}: {len(frames)} frames, expected {expected}")
    return frames


def pool(tokens: torch.Tensor, mode: str, grid: int = 4) -> torch.Tensor:
    """(batch, tokens, dim) -> (batch, latent).

    `mean` pools everything to 1024-d and throws away layout. `grid` keeps a
    coarse `grid` x `grid` spatial map, which is what a readout needs if it has to
    localise the block to centimetres.
    """
    if mode == "mean":
        return tokens.mean(dim=1)
    batch, count, dim = tokens.shape
    side = round((count / max(1, count // (16 * 16))) ** 0.5)
    time = count // (side * side)
    if side % grid or time * side * side != count:
        raise ValueError(f"{count} tokens do not split into {time} x {side} x {side} with a {grid}-cell grid")
    cells = tokens.reshape(batch, time, grid, side // grid, grid, side // grid, dim)
    return cells.mean(dim=(1, 3, 5)).reshape(batch, grid * grid * dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_wm"))
    ap.add_argument("--pool", default="mean",
                    help="comma-separated poolings to cache from one encoder pass: `mean`, or "
                         "`gridN` for an N x N spatial map. The first is written to --out and the "
                         "rest to `window_embeddings_<spec>.pt`. Encoding is the expensive half, "
                         "so caching several poolings together costs almost nothing extra. Mean "
                         "pooling discards layout, which is fine for one object and not fine when "
                         "a readout has to bind a heading to a particular block.")
    ap.add_argument("--grid", type=int, default=4, help="spatial cells per side for a bare `grid` spec")
    ap.add_argument("--model", type=Path, default=Path("models/vjepa2-vitl-fpc64-256"))
    ap.add_argument("--out", type=Path, default=None, help="default: <data>/window_embeddings.pt")
    ap.add_argument("--refresh", action="store_true", help="re-encode clips already in the cache")
    ap.add_argument("--windows", type=int, default=None,
                    help="encode only the first N windows instead of every one. The state world "
                         "model conditions on the pre-execution observation window alone, so N=1 "
                         "costs a fraction of the full pass; the latent-dynamics rollout needs all "
                         "of them, so leave this unset for that corpus.")
    args = ap.parse_args()
    specs = [spec.strip() for spec in str(args.pool).split(",") if spec.strip()]
    poolings = []
    for spec in specs:
        if spec == "mean":
            poolings.append((spec, "mean", 0))
        elif spec == "grid":
            poolings.append((f"grid{args.grid}", "grid", args.grid))
        elif spec.startswith("grid") and spec[4:].isdigit():
            poolings.append((spec, "grid", int(spec[4:])))
        else:
            raise SystemExit(f"unknown pooling {spec!r}; expected `mean`, `grid`, or `gridN`")
    if not poolings:
        raise SystemExit("--pool needs at least one pooling")

    out = args.out or args.data / "window_embeddings.pt"
    manifest = json.loads((args.data / "manifest.json").read_text())
    records = [json.loads(line) for line in (args.data / "records.jsonl").open()]
    window, count = manifest["window_frames"], manifest["windows"]
    if args.windows is not None:
        if not 1 <= args.windows <= count:
            raise SystemExit(f"--windows {args.windows} outside 1..{count} for this corpus")
        count = args.windows

    paths = [out if index == 0 else args.data / f"window_embeddings_{name}.pt"
             for index, (name, _, _) in enumerate(poolings)]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    caches = [torch.load(path, weights_only=False) if path.exists() and not args.refresh else {}
              for path in paths]
    # An episode is missing if any requested pooling lacks it, so one pass fills them all.
    missing = [r for r in records if any(r["episode_id"] not in cache for cache in caches)]
    if missing:
        processor = AutoVideoProcessor.from_pretrained(args.model, local_files_only=True)
        encoder = AutoModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
        started = time.time()
        with torch.inference_mode():
            for done, record in enumerate(missing, 1):
                frames = clip_frames(args.data / record["observation"]["frames_path"], manifest["frames_total"])
                batch = torch.cat([processor(frames[k * window:(k + 1) * window], return_tensors="pt")["pixel_values_videos"]
                                   for k in range(count)]).to(device)
                tokens = encoder(pixel_values_videos=batch).last_hidden_state
                for cache, (_, mode, grid) in zip(caches, poolings):
                    cache[record["episode_id"]] = pool(tokens, mode, grid).float().cpu()
                if done % 25 == 0 or done == len(missing):
                    rate = (time.time() - started) / done
                    print(f"embedded {done}/{len(missing)} ({rate:.2f}s/episode, {rate * (len(missing) - done) / 60:.1f} min left)", flush=True)
                if done % 200 == 0:
                    for cache, path in zip(caches, paths):
                        torch.save(cache, path)
        for cache, path in zip(caches, paths):
            torch.save(cache, path)
    print(json.dumps([{"pooling": name, "episodes": len(cache),
                       "per_episode_shape": tuple(next(iter(cache.values())).shape), "path": str(path)}
                      for (name, _, _), cache, path in zip(poolings, caches, paths)], indent=2))


if __name__ == "__main__":
    main()
