"""Generate schema-3 UR5e latent-transition episodes: prelude, skill, per-frame tracks."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import imageio.v3 as iio
import mujoco
import numpy as np
from waddle_wm.sim.env import FRAMES_TOTAL, PRELUDE_FRAMES, WINDOW_FRAMES, TabletopEnv
from waddle_wm.sim import relling_scene as scene
from waddle_wm.sim.env import pick_place_trace

DEFAULT_OUT = Path("data/ur5e_wm")
BAD_TARGET_PROB, BAD_GRASP_PROB = 0.45, 0.4
GOOD_GRASP_M, BAD_GRASP_M = (0.000, 0.012), (0.024, 0.042)  # lateral grasp error; the 36 mm block is lost past ~28 mm

# `--oriented`: the corpus a yaw-aware plan encoding needs. A cube grasped from above is the
# same object at every wrist heading, so a corpus of cubes and free wrists can only ever fit a
# model whose yaw dimensions are constant — which is exactly the orientation-blind checkpoint
# #35 is about. Here the source block is a rectangle, it spawns rotated, and every episode
# commands a heading: sometimes the block's own, sometimes deliberately across it. The outcome
# then depends on the heading, so the fitted model has something to learn from it.
ORIENTED_SIZE = (0.030, 0.011, 0.018)      # 60 x 22 mm footprint, 36 mm tall
ALIGNED_GRASP_PROB = 0.5                    # how often the commanded heading is the block's own
YAW_PHASES = ("approach", "descend", "lift")
# The cube corpus's lateral errors are scaled to a 36 mm block. A 22 mm-wide rectangle is lost
# at less than half that, so the oriented corpus scales them down; otherwise every episode
# misses and the heading explains nothing.
ORIENTED_GOOD_GRASP_M, ORIENTED_BAD_GRASP_M = (0.000, 0.005), (0.013, 0.030)


def wrap_yaw_deg(angle: float) -> float:
    """Fold a heading into the +-90 degrees the program schema allows; the jaws are symmetric."""
    return (float(angle) + 90.0) % 180.0 - 90.0


def xy(value):
    x, y = value.split(",", 1)
    return float(x), float(y)

def sample_params(env, rng):
    """A plan: where to place, and how well aimed the grasp is."""
    target = np.array(env.state()["target_pos"], dtype=float)
    if rng.random() < BAD_TARGET_PROB:
        target += [-float(rng.uniform(0.14, 0.20)), float(rng.uniform(-0.06, 0.06))]
    low, high = BAD_GRASP_M if rng.random() < BAD_GRASP_PROB else GOOD_GRASP_M
    angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(low, high)
    return {"target_xy": target.tolist(), "grasp_offset_xy": [radius * np.cos(angle), radius * np.sin(angle)]}


def orient_source(env, source: str, block_yaw_deg: float) -> None:
    """Make `source` an elongated block lying at `block_yaw_deg`, and the others plain cubes."""
    for name in scene.BLOCK_NAMES:
        size = ORIENTED_SIZE if name == source else (scene.BLOCK_HALF,) * 3
        env.model.geom_size[env.model.geom(f"{name}_geom").id] = size
    mujoco.mj_setConst(env.model, env.data)
    yaw = math.radians(block_yaw_deg)
    env.data.joint(f"{source}_free").qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    mujoco.mj_forward(env.model, env.data)


def mixed_episode(env, rng, index, oriented: bool = False):
    """Balanced source/destination coverage with both grasp and placement failures.

    With `oriented`, the source block is rotated and rectangular and the trace pins a wrist
    heading through the approach, the descent, and the lift — the heading is part of the action
    and reaches the world model's plan vector as `plan_encoding` v2's yaw terms.
    """
    source = scene.BLOCK_NAMES[index % len(scene.BLOCK_NAMES)]
    destinations = ("green_pad", *(name for name in scene.BLOCK_NAMES if name != source))
    destination = destinations[(index // len(scene.BLOCK_NAMES)) % len(destinations)]
    positions = env.sample_blocks()
    env.reset(blocks=positions)
    block_yaw_deg = grasp_yaw_deg = None
    if oriented:
        block_yaw_deg = float(rng.uniform(-90.0, 90.0))
        orient_source(env, source, block_yaw_deg)
        grasp_yaw_deg = (block_yaw_deg if rng.random() < ALIGNED_GRASP_PROB
                         else wrap_yaw_deg(block_yaw_deg + float(rng.uniform(20.0, 160.0))))
    source_xy = np.asarray(positions[source][:2])
    target = (np.asarray(env.state()["target_pos"]) if destination == "green_pad"
              else np.asarray(positions[destination][:2]))
    if rng.random() < BAD_TARGET_PROB:
        angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(0.06, 0.18)
        target = target + radius * np.array([np.cos(angle), np.sin(angle)])
    target = np.clip(target, (0.24, -0.40), (0.68, 0.45))
    bad_grasp = rng.random() < BAD_GRASP_PROB
    if oriented:
        low, high = ORIENTED_BAD_GRASP_M if bad_grasp else ORIENTED_GOOD_GRASP_M
    else:
        low, high = BAD_GRASP_M if bad_grasp else GOOD_GRASP_M
    angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(low, high)
    offset = radius * np.array([np.cos(angle), np.sin(angle)])
    trace = pick_place_trace(source_xy, target, offset)
    if destination != "green_pad":
        for entry in trace:
            if entry["phase"] == "place":
                entry["target"][2] = positions[destination][2] + 2 * scene.BLOCK_HALF
    if grasp_yaw_deg is not None:
        for entry in trace:
            if entry["phase"] in YAW_PHASES:
                entry["yaw"] = math.radians(grasp_yaw_deg)
    params = {"object": source, "destination": destination,
              "target_xy": target.tolist(), "grasp_offset_xy": offset.tolist(),
              **({} if not oriented else {"block_yaw_deg": block_yaw_deg,
                                          "grasp_yaw_deg": grasp_yaw_deg,
                                          "source_block_size": list(ORIENTED_SIZE)})}
    return env.run_trace(trace, frames_total=FRAMES_TOTAL, prelude_frames=PRELUDE_FRAMES,
                         skill="pick_place", params=params, block=source, destination=destination)

def phase_frames(records):
    """Median executed duration of each phase, used to compile plans without executing."""
    spans = {}
    for record in records:
        if record["split"] != "train":
            continue
        for entry in record["skill"]["trace"]:
            spans.setdefault(entry["phase"], []).append(entry["frames"][1] - entry["frames"][0] + 1)
    return {phase: float(np.median(lengths)) for phase, lengths in spans.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000); ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--size", type=int, default=256); ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--append", action="store_true", help="add episodes without replacing existing records")
    ap.add_argument("--mixed", action="store_true", help="schema-4 corpus spanning every block and pad/block destinations")
    ap.add_argument("--oriented", action="store_true",
                    help="mixed corpus whose source block is rectangular, rotated, and grasped at a "
                         "commanded wrist heading; required for a yaw-aware world-model checkpoint")
    ap.add_argument("--block-spawn-low", type=xy, default=(0.30, -0.26), help="x,y lower bound for red block sampling")
    ap.add_argument("--block-spawn-high", type=xy, default=(0.46, -0.10), help="x,y upper bound for red block sampling")
    args = ap.parse_args(); clips = args.out / "clips"; clips.mkdir(parents=True, exist_ok=True)
    records_path = args.out / "records.jsonl"
    records = [json.loads(line) for line in records_path.open()] if args.append and records_path.exists() else []
    start = len(records)
    env = TabletopEnv(width=args.size, height=args.size, fps=args.fps, seed=args.seed,
                      block_spawn_low=args.block_spawn_low, block_spawn_high=args.block_spawn_high)
    rng = np.random.default_rng(args.seed + 1)
    home = env.home_waypoint()
    for i in range(args.episodes):
        if args.mixed or args.oriented:
            for _ in range(20):
                try:
                    ep = mixed_episode(env, rng, start + i, oriented=args.oriented)
                    break
                except RuntimeError as error:
                    if "IK failed" not in str(error):
                        raise
            else:
                raise RuntimeError(f"could not sample a reachable mixed episode for index {start + i}")
        else:
            env.reset(env.sample_scene())
            ep = env.run_skill("pick_place", sample_params(env, rng))
        index = start + i; name = f"ur5e_{index:04d}.mp4"; iio.imwrite(clips / name, ep.frames, fps=args.fps, codec="libx264")
        split = "train" if index % 20 < 14 else ("val" if index % 20 < 17 else "test")
        records.append({"schema_version": 4 if (args.mixed or args.oriented) else 3, "episode_id": f"ur5e_{index:04d}", "seed": args.seed, "split": split,
            "observation": {"camera": "demo", "frames_path": f"clips/{name}", "width": args.size, "height": args.size, "fps": args.fps,
                            "frames_total": len(ep.frames), "prelude_frames": PRELUDE_FRAMES, "window_frames": WINDOW_FRAMES,
                            "frame_times_s": ep.frame_times},
            "skill": {"name": ep.skill, "params": ep.params, "trace": ep.skill_trace}, "tracks": ep.tracks,
            "state_before": ep.state_before, "state_after": ep.state_after,
            "outcome": {"success": ep.success, "failure_mode": ep.failure_mode}})
        if (i + 1) % 25 == 0 or i + 1 == args.episodes:
            print(f"{i + 1}/{args.episodes}: {'success' if ep.success else ep.failure_mode}", flush=True)
            with records_path.open("w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
    with records_path.open("w") as f:
        for record in records: f.write(json.dumps(record) + "\n")
    mixed = args.mixed or args.oriented
    (args.out / "manifest.json").write_text(json.dumps({
        "schema_version": 4 if mixed else 3, "episodes": len(records), "fps": args.fps, "frames_total": FRAMES_TOTAL,
        "prelude_frames": PRELUDE_FRAMES, "window_frames": WINDOW_FRAMES, "windows": FRAMES_TOTAL // WINDOW_FRAMES,
        "block_spawn_low": ((0.30, -0.28) if mixed else args.block_spawn_low),
        "block_spawn_high": ((0.68, -0.08) if mixed else args.block_spawn_high),
        "commanded_grasp_yaw": bool(args.oriented),
        "home_waypoint": home, "phase_frames": phase_frames(records),
        **({"block_names": list(scene.BLOCK_NAMES), "destinations": ["green_pad", *scene.BLOCK_NAMES]}
           if mixed else {})}, indent=2))
    print(f"wrote {len(records)} UR5e episodes to {args.out}")

if __name__ == "__main__": main()
