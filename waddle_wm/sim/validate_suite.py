"""Merge the task-suite shards into one corpus and refuse it if any check fails.

Generation is sharded across processes, so the corpus does not exist until this runs: it
concatenates `records_*.jsonl`, recomputes `phase_frames` over the merged train split, writes
`records.jsonl` and `manifest.json`, and then checks the result.

The checks that matter here and are not in `validate_dataset`:

* **labels are reproducible.** Every episode's subtask outcomes are recomputed from its
  per-frame tracks and compared to what was recorded. A label that cannot be rederived from
  the trajectory is a label nobody can audit.
* **yaw is informative and causal.** The commanded headings must vary, and the lift rate must
  fall as grasp misalignment grows. A corpus where it does not is one where the yaw dimensions
  are decoration, which is the exact failure this whole corpus exists to avoid.
* **no scene leaks across splits.** Scenes are keyed by the spawn layout *and* headings, so two
  episodes with the same positions at different headings are correctly counted as different.

    uv run python -m waddle_wm.sim.validate_suite --data data/ur5e_wm_suite
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from waddle_wm.sim.generate_suite import FAMILIES, FRAMES_TOTAL, WINDOWS, phase_frames, subtask_outcomes


def assign_splits(records: list[dict]) -> None:
    """Stratify train/val/test within each family, in place.

    The generator's own `index % 20` split cannot be used: the family cycles with period 4 and
    the split with period 20, and 4 divides 20, so the two are perfectly aliased — every val
    episode lands on a position whose family is one of three, and `stack` never appears in val
    at all. Splitting within each family instead gives all four families the same 70/15/15 in
    every split. Assignment is deterministic in episode-id order and happens before any model
    sees the corpus, so it is a stratified split, not a tuned one.
    """
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for record in sorted(records, key=lambda record: record["episode_id"]):
        by_family[record["family"]].append(record)
    for group in by_family.values():
        for position, record in enumerate(group):
            record["split"] = ("train" if position % 20 < 14 else
                               "val" if position % 20 < 17 else "test")


def merge(data: Path) -> tuple[list[dict], dict]:
    """Concatenate the shards in episode order and rebuild the corpus manifest."""
    shards = sorted(data.glob("records_*.jsonl"))
    if not shards:
        raise SystemExit(f"no records_*.jsonl shards in {data}")
    records = [json.loads(line) for shard in shards for line in shard.open()]
    records.sort(key=lambda record: record["episode_id"])
    assign_splits(records)

    manifests = [json.loads(path.read_text()) for path in sorted(data.glob("manifest_*.json"))]
    base = dict(manifests[0])
    for key in ("frames_total", "prelude_frames", "window_frames", "windows", "schema_version",
                "block_size", "block_names"):
        values = {json.dumps(m[key]) for m in manifests}
        if len(values) != 1:
            raise SystemExit(f"shards disagree on {key}: {values}")
    base["episodes"] = len(records)
    base["families"] = sorted({record["family"] for record in records})
    base["shards"] = len(shards)
    base["splits_assigned"] = "stratified by family at merge; see validate_suite.assign_splits"
    base["phase_frames"] = phase_frames(records)
    base.pop("seed", None)
    (data / "records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (data / "manifest.json").write_text(json.dumps(base, indent=2))
    return records, base


def scene_key(record: dict) -> str:
    """A fingerprint of the initial scene: where every block spawned and how it was turned."""
    params = record["skill"]["params"]
    payload = json.dumps([params["spawn_positions"], params["block_yaws_deg"]], sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def misalignment_deg(subtask: dict) -> float:
    """How far the commanded heading is from the block's own, folded to 0..90."""
    delta = abs(subtask["grasp_yaw_deg"] - subtask["block_yaw_deg"]) % 180.0
    return min(delta, 180.0 - delta)


def check(records: list[dict], manifest: dict, data: Path, sample: int = 40) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []

    def report(ok: bool, message: str):
        results.append((bool(ok), message))

    versions = {record["schema_version"] for record in records}
    report(versions == {5}, f"schema_version: {sorted(versions)}")
    report(len(records) == manifest["episodes"],
           f"episodes: {len(records)} records, manifest says {manifest['episodes']}")

    grid = {(record["observation"]["frames_total"], record["observation"]["prelude_frames"],
             record["observation"]["window_frames"]) for record in records}
    report(grid == {(FRAMES_TOTAL, manifest["prelude_frames"], manifest["window_frames"])},
           f"frame grid (total, prelude, window): {sorted(grid)}")
    report(manifest["windows"] == WINDOWS, f"windows: {manifest['windows']}")

    lengths = {len(value) for record in records for value in record["tracks"].values()}
    report(lengths == {FRAMES_TOTAL}, f"per-frame track lengths: {sorted(lengths)}")

    ids = [record["episode_id"] for record in records]
    report(len(set(ids)) == len(ids), f"duplicate episode ids: {len(ids) - len(set(ids))}")

    keys = collections.defaultdict(set)
    for record in records:
        keys[record["split"]].add(scene_key(record))
    total_scenes = sum(len(value) for value in keys.values())
    report(total_scenes == len(records), f"duplicate scenes: {len(records) - total_scenes}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = keys[a] & keys[b]
        report(not shared, f"{a} n {b} shared scenes: {len(shared)}")

    splits = collections.Counter(record["split"] for record in records)
    report(set(splits) == {"train", "val", "test"}, f"splits: {dict(splits)}")

    # Every family present in every split, and none of them degenerate in either direction.
    for family in manifest["families"]:
        report(family in FAMILIES, f"known family: {family}")
    for split in ("train", "val", "test"):
        for family in manifest["families"]:
            subset = [r for r in records if r["split"] == split and r["family"] == family]
            rate = float(np.mean([r["outcome"]["success"] for r in subset])) if subset else 0.0
            report(bool(subset) and 0.05 <= rate <= 0.95,
                   f"  {split}/{family}: n={len(subset)} success={rate:.3f}")

    # Labels have to be a function of the trajectory, not of the generator's memory.
    mismatched = []
    for record in records:
        recomputed = subtask_outcomes(record["tracks"], record["skill"]["params"]["subtasks"])
        stored = record["subtask_outcomes"]
        if [o["success"] for o in recomputed] != [o["success"] for o in stored]:
            mismatched.append(record["episode_id"])
    report(not mismatched, f"outcomes recomputable from tracks: {len(mismatched)} mismatched")

    # Yaw has to vary, and it has to matter.
    subtasks = [s for record in records for s in record["skill"]["params"]["subtasks"]]
    headings = np.array([s["grasp_yaw_deg"] for s in subtasks])
    report(float(headings.std()) > 10.0, f"commanded heading spread: std {headings.std():.1f} deg")
    aligned = [s["yaw_aligned"] for s in subtasks]
    report(0.2 <= float(np.mean(aligned)) <= 0.8, f"aligned grasps: {np.mean(aligned):.3f}")

    lifts = [(misalignment_deg(s), o["lifted"])
             for record in records
             for s, o in zip(record["skill"]["params"]["subtasks"], record["subtask_outcomes"])]
    near = [lifted for angle, lifted in lifts if angle <= 10.0]
    far = [lifted for angle, lifted in lifts if angle >= 60.0]
    report(bool(near) and bool(far) and np.mean(near) - np.mean(far) > 0.3,
           f"yaw is causal: lift {np.mean(near):.3f} aligned (n={len(near)}) vs "
           f"{np.mean(far):.3f} at >=60 deg (n={len(far)})")

    low, high = np.asarray(manifest["block_spawn_low"]), np.asarray(manifest["block_spawn_high"])
    spawns = np.array([xy[:2] for record in records
                       for xy in record["skill"]["params"]["spawn_positions"].values()])
    report(bool((spawns >= low - 1e-6).all() and (spawns <= high + 1e-6).all()),
           f"spawn box {list(low)} .. {list(high)}: observed "
           f"{list(np.round(spawns.min(0), 4))} .. {list(np.round(spawns.max(0), 4))}")

    missing = [record["episode_id"] for record in records
               if not (data / record["observation"]["frames_path"]).exists()]
    report(not missing, f"missing clips: {len(missing)}")

    if not missing:
        import cv2
        rng = np.random.default_rng(0)
        picks = rng.choice(len(records), size=min(sample, len(records)), replace=False)
        bad = []
        for index in picks:
            record = records[int(index)]
            capture = cv2.VideoCapture(str(data / record["observation"]["frames_path"]))
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            capture.release()
            if len(frames) != FRAMES_TOTAL or np.asarray(frames).std() < 1.0:
                bad.append(record["episode_id"])
        report(not bad, f"decoded {len(picks)} sampled clips, {len(bad)} malformed or blank")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_wm_suite"))
    ap.add_argument("--sample", type=int, default=40, help="clips to decode as a spot check")
    args = ap.parse_args()

    records, manifest = merge(args.data)
    results = check(records, manifest, args.data, args.sample)
    for ok, message in results:
        print(f"{'PASS' if ok else 'FAIL'}  {message}")
    failed = [message for ok, message in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        raise SystemExit(f"{len(failed)} checks failed")


if __name__ == "__main__":
    main()
