"""Generate the MuJoCo skill-transition dataset.

Runs randomized scenes through the three skills, deliberately including bad
skill traces, and writes one mp4 clip plus one JSON record per episode.

    uv run python -m waddle_wm.sim.generate_dataset --episodes 120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from waddle_wm.sim.env import CLOSED_WIDTH, TARGET_XY, TabletopEnv

DEFAULT_OUT = Path("data/mujoco_tabletop")


def sample_plan(rng: np.random.Generator) -> tuple[str, dict]:
    """Pick a skill and its parameters; ~45% of plans are deliberately bad."""
    skill = str(rng.choice(["top_grasp", "side_grasp", "align"]))
    bad = rng.random() < 0.45

    if skill == "top_grasp":
        if not bad:
            return skill, {
                "offset_x": float(rng.uniform(-0.004, 0.004)),
                "offset_y": float(rng.uniform(-0.004, 0.004)),
                "grip_width": CLOSED_WIDTH,
            }
        mode = rng.integers(0, 2)
        if mode == 0:  # aimed off the block
            return skill, {
                "offset_y": float(rng.choice([-1, 1]) * rng.uniform(0.028, 0.05)),
                "grip_width": CLOSED_WIDTH,
            }
        return skill, {"grip_width": float(rng.uniform(0.048, 0.068))}  # too slack

    if skill == "side_grasp":
        if not bad:
            return skill, {"side": -1.0, "grip_width": CLOSED_WIDTH}
        mode = rng.integers(0, 2)
        if mode == 0:  # approaches straight across the bin wall
            return skill, {"side": 1.0, "grip_width": CLOSED_WIDTH}
        return skill, {"side": -1.0, "grip_width": float(rng.uniform(0.048, 0.068))}

    if not bad:
        return skill, {"push_to_x": float(TARGET_XY[0] + rng.uniform(-0.02, 0.02))}
    return skill, {"push_to_x": float(rng.choice([rng.uniform(-0.13, -0.06),
                                                  rng.uniform(-0.40, -0.32)]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", default="frontal")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    clips = args.out / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    env = TabletopEnv(camera=args.camera, width=args.size, height=args.size,
                      fps=args.fps, seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)

    records = []
    for i in range(args.episodes):
        env.reset(env.sample_scene())
        skill, params = sample_plan(rng)
        ep = env.run_skill(skill, params)

        name = f"{i:04d}_{skill}.mp4"
        iio.imwrite(clips / name, ep.frames, fps=args.fps, codec="libx264")

        # deterministic 70/15/15 split by episode index
        split = "train" if i % 20 < 14 else ("val" if i % 20 < 17 else "test")
        records.append(
            {
                "id": f"{i:04d}",
                "split": split,
                "clip": f"clips/{name}",
                "n_frames": int(len(ep.frames)),
                "fps": args.fps,
                "camera": args.camera,
                "skill": ep.skill,
                "params": ep.params,
                "skill_trace": ep.skill_trace,
                "state_before": ep.state_before,
                "state_after": ep.state_after,
                "success": ep.success,
                "failure_mode": ep.failure_mode,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.episodes} episodes")

    with (args.out / "records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    _summarize(records)
    print(f"\nwrote {len(records)} episodes to {args.out}")


def _summarize(records: list[dict]) -> None:
    print("\nsplit      n   success_rate")
    for split in ("train", "val", "test"):
        rs = [r for r in records if r["split"] == split]
        rate = sum(r["success"] for r in rs) / max(1, len(rs))
        print(f"{split:6s} {len(rs):4d}   {rate:.2f}")

    print("\nskill        n   success_rate")
    for skill in ("top_grasp", "side_grasp", "align"):
        rs = [r for r in records if r["skill"] == skill]
        rate = sum(r["success"] for r in rs) / max(1, len(rs))
        print(f"{skill:10s} {len(rs):4d}   {rate:.2f}")

    modes: dict[str, int] = {}
    for r in records:
        key = r["failure_mode"] or "success"
        modes[key] = modes.get(key, 0) + 1
    print("\noutcome")
    for k, v in sorted(modes.items(), key=lambda kv: -kv[1]):
        print(f"  {k:26s} {v}")


if __name__ == "__main__":
    main()
