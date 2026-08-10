"""Check that the corpus validator fails on the corruptions it exists to catch.

    uv run python -m waddle_wm.sim.test_validate_dataset
"""
from __future__ import annotations
import copy, json, tempfile
from pathlib import Path
import imageio.v3 as iio
import numpy as np
from waddle_wm.sim import validate_dataset as V
from waddle_wm.sim.env import FRAMES_TOTAL, PRELUDE_FRAMES, TRACK_KEYS, WINDOW_FRAMES

LOW, HIGH = (0.30, -0.26), (0.46, -0.10)
OUTCOMES = [(True, None), (False, "missed"), (False, "target_miss")]


def make_records(n=60, low=LOW, high=HIGH):
    rng = np.random.default_rng(0)
    records = []
    for i in range(n):
        block = rng.uniform(low, high)
        success, mode = OUTCOMES[i % 3]
        records.append({
            "schema_version": 3, "episode_id": f"ur5e_{i:04d}", "seed": 0,
            "split": "train" if i % 20 < 14 else ("val" if i % 20 < 17 else "test"),
            "observation": {"camera": "demo", "frames_path": f"clips/ur5e_{i:04d}.mp4", "width": 64, "height": 64,
                            "fps": 10, "frames_total": FRAMES_TOTAL, "prelude_frames": PRELUDE_FRAMES,
                            "window_frames": WINDOW_FRAMES},
            "skill": {"name": "pick_place", "params": {"target_xy": [0.1 + 0.001 * i, 0.4],
                                                       "grasp_offset_xy": [0.001 * i, 0.0]}, "trace": []},
            "tracks": {key: [0.0] * FRAMES_TOTAL for key in TRACK_KEYS},
            "state_before": {"block_pos": [*block, 0.02], "target_pos": [0.1, 0.4]}, "state_after": {},
            "outcome": {"success": success, "failure_mode": mode}})
    return records


def write_corpus(root, records, low=LOW, high=HIGH):
    root.mkdir(parents=True, exist_ok=True)
    (root / "clips").mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    with (root / "records.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
            obs = record["observation"]
            frames = rng.integers(0, 255, (obs["frames_total"], obs["height"], obs["width"], 3), dtype=np.uint8)
            iio.imwrite(root / obs["frames_path"], frames, fps=obs["fps"], codec="libx264")
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 3, "episodes": len(records), "fps": 10, "frames_total": FRAMES_TOTAL,
        "prelude_frames": PRELUDE_FRAMES, "window_frames": WINDOW_FRAMES, "windows": FRAMES_TOTAL // WINDOW_FRAMES,
        "block_spawn_low": list(low), "block_spawn_high": list(high)}))
    return root


def run(root):
    """Every check, returning the messages that failed."""
    records = [json.loads(line) for line in (root / "records.jsonl").open()]
    manifest = json.loads((root / "manifest.json").read_text())
    failures = []

    def report(message, ok):
        if not ok:
            failures.append(message)

    V.check_schema(records, manifest, report)
    V.check_spawn(records, manifest, report)
    V.check_splits(records, report)
    V.check_leakage(records, report)
    V.check_rendering(records, root, 8, 0, report)
    return failures


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clean = make_records()
        assert not run(write_corpus(tmp / "clean", clean)), run(write_corpus(tmp / "clean", clean))

        appended = make_records(30, low=(0.34, -0.22), high=(0.42, -0.14)) + [
            dict(record, episode_id=f"ur5e_{i + 30:04d}", observation=dict(record["observation"],
                 frames_path=f"clips/ur5e_{i + 30:04d}.mp4"))
            for i, record in enumerate(make_records(30))]
        assert any("spawn spread" in f for f in run(write_corpus(tmp / "appended", appended))), "append went undetected"

        strayed = make_records(60, low=(0.20, -0.30), high=(0.46, -0.10))  # wider than the manifest declares
        assert any("spawn box" in f for f in run(write_corpus(tmp / "strayed", strayed))), "out-of-box spawn undetected"

        leaked = copy.deepcopy(clean)
        for record in leaked:
            if record["split"] == "test":
                record["skill"]["params"] = copy.deepcopy(leaked[0]["skill"]["params"])
                record["state_before"]["block_pos"] = list(leaked[0]["state_before"]["block_pos"])
        assert any("shared scenes" in f for f in run(write_corpus(tmp / "leaked", leaked))), "leakage undetected"

        skewed = copy.deepcopy(clean)
        for record in skewed:
            record["outcome"] = {"success": True, "failure_mode": None}
        assert any("success=1.000" in f for f in run(write_corpus(tmp / "skewed", skewed))), "outcome skew undetected"

        short = copy.deepcopy(clean)
        short[0]["tracks"]["phase"] = short[0]["tracks"]["phase"][:-1]
        assert any("track lengths" in f for f in run(write_corpus(tmp / "short", short))), "short track undetected"

        blank = copy.deepcopy(clean)
        write_corpus(tmp / "blank", blank)
        for record in blank[:8]:
            obs = record["observation"]
            iio.imwrite(tmp / "blank" / obs["frames_path"],
                        np.zeros((obs["frames_total"], obs["height"], obs["width"], 3), np.uint8), fps=obs["fps"],
                        codec="libx264")
        assert any("malformed or blank" in f for f in run(tmp / "blank")), "blank clips undetected"

        missing = copy.deepcopy(clean)
        write_corpus(tmp / "missing", missing)
        (tmp / "missing" / missing[0]["observation"]["frames_path"]).unlink()
        assert any("missing clips" in f for f in run(tmp / "missing")), "missing clip undetected"
    print("validate_dataset check passed: clean corpus passes, 7 corruptions caught")


if __name__ == "__main__":
    main()
