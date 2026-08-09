"""Generate physical UR5e skill-transition clips and records."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import imageio.v3 as iio
import numpy as np
from waddle_wm.sim.env import TabletopEnv

DEFAULT_OUT = Path("data/ur5e_tabletop")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120); ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--size", type=int, default=256); ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--append", action="store_true", help="add episodes without replacing existing records")
    args = ap.parse_args(); clips = args.out / "clips"; clips.mkdir(parents=True, exist_ok=True)
    records_path = args.out / "records.jsonl"
    records = [json.loads(line) for line in records_path.open()] if args.append and records_path.exists() else []
    start = len(records)
    env, rng = TabletopEnv(width=args.size, height=args.size, fps=args.fps, seed=args.seed), np.random.default_rng(args.seed + 1)
    for i in range(args.episodes):
        env.reset(env.sample_scene())
        target = np.array(env.state()["target_pos"])
        bad = rng.random() < 0.45
        if bad: target += [-float(rng.uniform(0.14, 0.20)), float(rng.uniform(-0.06, 0.06))]
        ep = env.run_skill("pick_place", {"target_xy": target.tolist()})
        index = start + i; name = f"ur5e_{index:04d}.mp4"; iio.imwrite(clips / name, ep.frames, fps=args.fps, codec="libx264")
        split = "train" if index % 20 < 14 else ("val" if index % 20 < 17 else "test")
        records.append({"schema_version": 2, "episode_id": f"ur5e_{index:04d}", "seed": args.seed, "split": split,
            "observation": {"camera": "demo", "frames_path": f"clips/{name}", "frame_times_s": ep.frame_times, "width": args.size, "height": args.size, "fps": args.fps},
            "skill": {"name": ep.skill, "params": ep.params, "trace": ep.skill_trace}, "state_before": ep.state_before, "state_after": ep.state_after,
            "outcome": {"success": ep.success, "failure_mode": ep.failure_mode}})
        print(f"{i + 1}/{args.episodes}: {'success' if ep.success else ep.failure_mode}")
    with (args.out / "records.jsonl").open("w") as f:
        for record in records: f.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} UR5e episodes to {args.out}")

if __name__ == "__main__": main()
