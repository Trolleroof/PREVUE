"""Validate a generated corpus: schema, splits, outcome balance, leakage, rendering."""
from __future__ import annotations
import argparse, collections, json, random
from pathlib import Path
import imageio.v3 as iio
import numpy as np
from waddle_wm.sim.env import FRAMES_TOTAL, PRELUDE_FRAMES, WINDOW_FRAMES

SPLITS = ("train", "val", "test")
FAILURE_MODES = ("missed", "target_miss")


def scene_key(record):
    """What makes an episode unique: where the block was, and the plan aimed at it."""
    params = record["skill"]["params"]
    return (tuple(np.round(record["state_before"]["block_pos"][:2], 6)),
            tuple(np.round(params["target_xy"], 6)), tuple(np.round(params["grasp_offset_xy"], 6)))


def check_schema(records, manifest, report):
    versions = {record["schema_version"] for record in records}
    report(f"schema_version: {sorted(versions)}", versions == {manifest["schema_version"]})
    report(f"episodes: {len(records)} records, manifest says {manifest['episodes']}",
           len(records) == manifest["episodes"])
    grids = {(record["observation"]["frames_total"], record["observation"]["prelude_frames"],
              record["observation"]["window_frames"]) for record in records}
    report(f"frame grid (total, prelude, window): {sorted(grids)}",
           grids == {(FRAMES_TOTAL, PRELUDE_FRAMES, WINDOW_FRAMES)})
    lengths = {len(track) for record in records for track in record["tracks"].values()}
    report(f"per-frame track lengths: {sorted(lengths)}", lengths == {FRAMES_TOTAL})


def check_spawn(records, manifest, report):
    """Every block must land inside the manifest's spawn box, not just on average."""
    if "block_spawn_low" not in manifest:  # pre-widening manifests recorded no box
        report("manifest predates --block-spawn-low/high; cannot check the spawn box", False)
        return
    low, high = np.array(manifest["block_spawn_low"]), np.array(manifest["block_spawn_high"])
    blocks = np.array([record["state_before"]["block_pos"][:2] for record in records])
    inside = ((blocks >= low - 1e-6) & (blocks <= high + 1e-6)).all(axis=1)
    report(f"block spawn box {low.tolist()} .. {high.tolist()}: "
           f"observed {blocks.min(0).round(4).tolist()} .. {blocks.max(0).round(4).tolist()}", bool(inside.all()))
    half = len(records) // 2
    first, second = np.ptp(blocks[:half], axis=0), np.ptp(blocks[half:], axis=0)
    report(f"spawn spread, first half {first.round(4).tolist()} vs second half {second.round(4).tolist()} "
           "(a narrow first half means episodes were appended to an older corpus)",
           bool((abs(first - second) < 0.25 * (high - low)).all()))


def check_splits(records, report):
    counts = collections.Counter(record["split"] for record in records)
    report(f"splits: {dict(counts)}", set(counts) == set(SPLITS) and all(counts[s] for s in SPLITS))
    for split in SPLITS:
        in_split = [record for record in records if record["split"] == split]
        modes = collections.Counter(record["outcome"]["failure_mode"] or "success" for record in in_split)
        rate = modes["success"] / len(in_split)
        report(f"  {split}: n={len(in_split)} success={rate:.3f} " +
               " ".join(f"{mode}={modes[mode] / len(in_split):.3f}" for mode in FAILURE_MODES),
               0.2 < rate < 0.8 and all(modes[mode] for mode in FAILURE_MODES))


def check_leakage(records, report):
    ids = [record["episode_id"] for record in records]
    report(f"duplicate episode ids: {len(ids) - len(set(ids))}", len(ids) == len(set(ids)))
    keys = [scene_key(record) for record in records]
    report(f"duplicate scenes: {len(keys) - len(set(keys))}", len(keys) == len(set(keys)))
    by_split = collections.defaultdict(set)
    for record, key in zip(records, keys):
        by_split[record["split"]].add(key)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = by_split[a] & by_split[b]
        report(f"  {a} n {b} shared scenes: {len(shared)}", not shared)


def check_rendering(records, root, sample, seed, report):
    missing = [r["episode_id"] for r in records if not (root / r["observation"]["frames_path"]).exists()]
    report(f"missing clips: {len(missing)}" + (f" e.g. {missing[:3]}" if missing else ""), not missing)
    chosen = random.Random(seed).sample(records, min(sample, len(records)))
    bad = []
    for record in chosen:
        obs = record["observation"]
        video = iio.imread(root / obs["frames_path"])
        if video.shape != (obs["frames_total"], obs["height"], obs["width"], 3) or float(video.std()) < 1.0:
            bad.append((record["episode_id"], video.shape, round(float(video.std()), 2)))
    report(f"decoded {len(chosen)} sampled clips, {len(bad)} malformed or blank" +
           (f": {bad[:3]}" if bad else ""), not bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_wm"))
    ap.add_argument("--render-sample", type=int, default=50, help="clips to decode; 0 skips the render check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    records = [json.loads(line) for line in (args.data / "records.jsonl").open()]
    manifest = json.loads((args.data / "manifest.json").read_text())
    failures = []

    def report(message, ok):
        print(f"{'PASS' if ok else 'FAIL'}  {message}", flush=True)
        if not ok:
            failures.append(message)

    print(f"# {args.data}")
    check_schema(records, manifest, report)
    check_spawn(records, manifest, report)
    check_splits(records, report)
    check_leakage(records, report)
    if args.render_sample:
        check_rendering(records, args.data, args.render_sample, args.seed, report)
    print(f"\n{len(failures)} failed check(s)" if failures else "\nall checks passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
