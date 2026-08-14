"""Generate the schema-5 task suite: several pick-and-place task families, every grasp
oriented, on one uniform frame grid.

Why a new corpus rather than more of `generate_dataset --oriented`:

* **Yaw has to matter everywhere.** In the oriented corpus only the *source* block is
  rectangular, so a two-object task would have a yaw-sensitive first grasp and a
  yaw-blind second one. Here every block is the same rectangle, so every grasp in every
  family is a decision about heading.
* **A task is more than one place.** `ordered_pad` and `ordered_stack` are two pick-and-
  places in one episode, which is what "place the blue block, then the red one" and
  "stack them" actually are. The second subtask's destination depends on where the first
  one *planned* to leave its block, not on where that block spawned — so a verifier has to
  read the plan as a sequence, not as one grasp and one place.
* **The outcome is per subtask.** `env` tracks a single block, which cannot express "the
  first place worked and the second missed". Outcomes here are recomputed per subtask from
  the per-frame tracks, and the episode succeeds only when every subtask does.

Everything else — the camera, the phase vocabulary, the 8-frame window grid, the
`records.jsonl` shape — is deliberately unchanged, so `windows.build`, `actions.compile_plan`
and `embed_windows` all read this corpus without modification.

    uv run python -m waddle_wm.sim.generate_suite --episodes 400 --out data/ur5e_wm_suite
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from waddle_wm.sim import relling_scene as scene
from waddle_wm.sim.env import (GRASP_Z, HOVER_Z, LIFT_THRESHOLD, PRELUDE_FRAMES, TARGET_RADIUS,
                               WINDOW_FRAMES, TabletopEnv, pick_place_trace)

SCHEMA_VERSION = 5
DEFAULT_OUT = Path("data/ur5e_wm_suite")

# 100 x 22 x 36 mm. The height matches the cube's, so a stack clears the same threshold.
#
# The length is the load-bearing choice. A 2F-85 opens to ~85 mm, so a block *longer* than the
# stroke cannot be grasped across its length at all, while the 22 mm width is trivial: the
# commanded heading now decides whether the grasp is possible, rather than merely tidier. At a
# misalignment of t degrees the jaws must span about 100*sin(t) + 22*cos(t) mm, which crosses
# the stroke near 40 degrees — so misalignment degrades smoothly into impossible, which is a
# learnable signal rather than a step. A 60 mm block, measured, fit the jaws at every heading
# and made yaw decorative.
SUITE_BLOCK_SIZE = (0.050, 0.011, 0.018)
JAW_STROKE_M = 0.085
BLOCK_HEIGHT = 2 * SUITE_BLOCK_SIZE[2]

# Two subtasks of eight phases each, plus the prelude, plus settling room. Measured worst
# case is 79 frames; 88 is 11 whole windows and leaves margin for a slow IK path.
FRAMES_TOTAL = 88
WINDOWS = FRAMES_TOTAL // WINDOW_FRAMES

SPAWN_LOW, SPAWN_HIGH = (0.30, -0.30), (0.68, -0.06)
SPAWN_SEPARATION = 0.14         # two 100 mm blocks, whatever their headings, cannot overlap
YAW_PHASES = ("approach", "descend", "lift")

# Lateral grasp error, scaled to the 22 mm jaw dimension rather than the cube's 36 mm.
GOOD_GRASP_M, BAD_GRASP_M = (0.000, 0.005), (0.013, 0.030)
GOOD_TARGET_M, BAD_TARGET_M = (0.000, 0.020), (0.060, 0.170)

FAMILIES = ("pad_place", "stack", "ordered_pad", "ordered_stack")
# Two-subtask families fail if either half fails, so their per-subtask fault rates are lower;
# otherwise the ordered families would be almost entirely negative and teach only the prior.
FAULT_RATES = {"pad_place": (0.38, 0.34), "stack": (0.38, 0.30),
               "ordered_pad": (0.22, 0.18), "ordered_stack": (0.22, 0.18)}

PAD_OFFSET_M = 0.070            # two blocks side by side on the pad, both still inside its radius

# What counts as stacked. The cube corpus used a 27 mm radius, which is tighter than this arm
# places: measured landing error is 12 mm median / 31 mm p90 against the commanded aim, and a
# stack inherits the support block's error too, so 27 mm rejected blocks visibly resting on
# top. These two say "the centre is over the support's footprint, and it is up on the support
# rather than beside it" — the support is 100 mm long, so half its length is the footprint
# bound, and half a block height is unambiguously up.
STACK_XY_TOL = 0.050
STACK_Z_MIN = BLOCK_HEIGHT / 2
ALIGNED_GRASP_PROB = 0.5
ACROSS_YAW_RANGE = (25.0, 155.0)


def wrap_yaw_deg(angle: float) -> float:
    """Fold a heading into +-90 degrees; the jaws are symmetric, so 180 degrees is the same grasp."""
    return (float(angle) + 90.0) % 180.0 - 90.0


def suite_env(**kwargs) -> TabletopEnv:
    """A tabletop whose three blocks are all the same rectangle."""
    return TabletopEnv(block_sizes={name: SUITE_BLOCK_SIZE for name in scene.BLOCK_NAMES}, **kwargs)


def sample_layout(rng) -> dict[str, list[float]]:
    """Three separated tabletop positions, each with its own spawn heading."""
    positions: dict[str, list[float]] = {}
    for name in scene.BLOCK_NAMES:
        for _ in range(200):
            xy = rng.uniform(SPAWN_LOW, SPAWN_HIGH)
            if all(np.linalg.norm(xy - np.asarray(other)[:2]) >= SPAWN_SEPARATION
                   for other in positions.values()):
                positions[name] = [*xy, SUITE_BLOCK_SIZE[2]]
                break
        else:
            raise RuntimeError("could not sample separated block positions")
    return positions


def apply_layout(env: TabletopEnv, positions: dict, yaws: dict[str, float]) -> None:
    """Place and rotate every block, then re-run kinematics so the render matches."""
    env.reset(blocks=positions)
    for name, yaw_deg in yaws.items():
        yaw = math.radians(yaw_deg)
        env.data.joint(f"{name}_free").qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    mujoco.mj_forward(env.model, env.data)


def sample_yaws(rng) -> dict[str, float]:
    return {name: float(rng.uniform(-90.0, 90.0)) for name in scene.BLOCK_NAMES}


def grasp_heading(rng, block_yaw_deg: float) -> tuple[float, bool]:
    """The commanded wrist heading: the block's own, or deliberately across it."""
    if rng.random() < ALIGNED_GRASP_PROB:
        return wrap_yaw_deg(block_yaw_deg), True
    across = block_yaw_deg + float(rng.uniform(*ACROSS_YAW_RANGE))
    return wrap_yaw_deg(across), False


def offset(rng, low: float, high: float) -> np.ndarray:
    angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(low, high)
    return radius * np.array([np.cos(angle), np.sin(angle)])


def plan_subtasks(rng, family: str, positions: dict, yaws: dict, pad_xy) -> list[dict]:
    """Which block goes where, how well aimed, and at what heading — before anything runs.

    Every destination here is what the *plan* believes, which for a second subtask means
    where the first subtask intended to leave its block. Nothing consults the simulator.
    """
    names = list(scene.BLOCK_NAMES)
    rng.shuffle(names)
    grasp_fault, target_fault = FAULT_RATES[family]
    pad = np.asarray(pad_xy, dtype=float)

    if family == "pad_place":
        goals = [(names[0], "green_pad", pad, 0.0)]
    elif family == "stack":
        support = names[1]
        goals = [(names[0], support, np.asarray(positions[support][:2]), positions[support][2])]
    elif family == "ordered_pad":
        # "place the blue block, then the red one" — two spots on the pad, far enough apart
        # that the second place is a distinct decision rather than a collision.
        direction = rng.uniform(0, 2 * np.pi)
        step = PAD_OFFSET_M * np.array([np.cos(direction), np.sin(direction)])
        goals = [(names[0], "green_pad", pad + step, 0.0),
                 (names[1], "green_pad", pad - step, 0.0)]
    else:   # ordered_stack: clear a spot on the pad, then stack the second block on the first
        goals = [(names[0], "green_pad", pad, 0.0),
                 (names[1], names[0], pad, SUITE_BLOCK_SIZE[2])]

    subtasks = []
    for source, destination, destination_xy, destination_z in goals:
        bad_grasp = rng.random() < grasp_fault
        bad_target = rng.random() < target_fault
        grasp_low, grasp_high = BAD_GRASP_M if bad_grasp else GOOD_GRASP_M
        target_low, target_high = BAD_TARGET_M if bad_target else GOOD_TARGET_M
        grasp_offset = offset(rng, grasp_low, grasp_high)
        target = np.clip(destination_xy + offset(rng, target_low, target_high),
                         (0.24, -0.40), (0.68, 0.45))
        grasp_yaw_deg, aligned = grasp_heading(rng, yaws[source])
        subtasks.append({
            "object": source, "destination": destination,
            "target_xy": target.tolist(),
            # Where the block must actually end up. For the pad that is the pad, not the
            # plan's aim point — a plan that aims 170 mm wide has missed the task, which is
            # the whole reason a target fault is a fault. For a block destination the goal
            # moves with the destination, so it is resolved at scoring time.
            "goal_xy": pad.tolist() if destination == "green_pad" else None,
            "place_z": float(destination_z + BLOCK_HEIGHT) if destination != "green_pad" else GRASP_Z,
            "grasp_offset_xy": grasp_offset.tolist(),
            "block_yaw_deg": float(yaws[source]), "grasp_yaw_deg": float(grasp_yaw_deg),
            "yaw_aligned": bool(aligned),
            "planted_grasp_fault": bool(bad_grasp), "planted_target_fault": bool(bad_target),
        })
    return subtasks


def subtask_trace(subtask: dict, source_xy) -> list[dict]:
    """One subtask's eight phases, with the wrist heading pinned through the grasp."""
    trace = pick_place_trace(source_xy, subtask["target_xy"], subtask["grasp_offset_xy"])
    for entry in trace:
        if entry["phase"] == "place":
            entry["target"][2] = subtask["place_z"]
        if entry["phase"] in YAW_PHASES:
            entry["yaw"] = math.radians(subtask["grasp_yaw_deg"])
    return trace


def build_trace(subtasks: list[dict], positions: dict) -> list[dict]:
    """Concatenate the subtasks into one waypoint program, as the plan sees it."""
    trace: list[dict] = []
    for subtask in subtasks:
        trace.extend(subtask_trace(subtask, np.asarray(positions[subtask["object"]][:2])))
    return trace


def subtask_outcomes(tracks: dict, subtasks: list[dict]) -> list[dict]:
    """Score each subtask from the per-frame tracks, not from the env's single tracked block.

    A subtask succeeds when its block was lifted clear of the table at some point *and*
    ended within its destination's radius — and, for a block destination, resting above it.
    Final positions are the settled ones, so a block knocked off by a later subtask counts
    as a failure of the subtask that was supposed to place it. That is the honest reading:
    the episode is one plan, and the plan left the block on the floor.
    """
    names = list(scene.BLOCK_NAMES)
    positions = np.asarray(tracks["all_block_pos"])          # (frames, blocks, 3)
    final = positions[-1]
    max_z = positions[..., 2].max(axis=0)

    outcomes = []
    for subtask in subtasks:
        index = names.index(subtask["object"])
        destination = subtask["destination"]
        if destination == "green_pad":
            goal_xy, radius = np.asarray(subtask["goal_xy"], dtype=float), TARGET_RADIUS
            resting = True
        else:
            support = final[names.index(destination)]
            goal_xy, radius = support[:2], STACK_XY_TOL
            resting = bool(final[index][2] >= support[2] + STACK_Z_MIN)
        lifted = bool(max_z[index] > LIFT_THRESHOLD)
        distance = float(np.linalg.norm(final[index][:2] - goal_xy))
        placed = bool(distance <= radius)
        success = bool(lifted and placed and resting)
        outcomes.append({
            "object": subtask["object"], "destination": destination, "success": success,
            "lifted": lifted, "placed": placed, "resting": resting,
            "final_distance_m": distance,
            "failure_mode": None if success else ("missed" if not lifted else
                                                  "not_stacked" if not resting else "target_miss"),
        })
    return outcomes


def episode_outcome(outcomes: list[dict]) -> dict:
    """The plan succeeds only if every subtask does; the failure is the first one that broke."""
    failed = next((index for index, outcome in enumerate(outcomes) if not outcome["success"]), None)
    if failed is None:
        return {"success": True, "failure_mode": None, "failed_subtask": None}
    return {"success": False, "failure_mode": outcomes[failed]["failure_mode"], "failed_subtask": failed}


def run_episode(env: TabletopEnv, rng, family: str) -> dict:
    """One episode: sample a layout, plan the family's subtasks, execute them as one attempt."""
    positions, yaws = sample_layout(rng), sample_yaws(rng)
    apply_layout(env, positions, yaws)
    pad_xy = np.asarray(env.state()["target_pos"], dtype=float)
    subtasks = plan_subtasks(rng, family, positions, yaws, pad_xy)
    trace = build_trace(subtasks, positions)
    last = subtasks[-1]
    params = {"family": family, "subtasks": subtasks,
              "object": subtasks[0]["object"], "destination": subtasks[0]["destination"],
              "target_xy": subtasks[0]["target_xy"],
              "grasp_offset_xy": subtasks[0]["grasp_offset_xy"],
              "block_yaws_deg": {name: float(value) for name, value in yaws.items()},
              "block_size": list(SUITE_BLOCK_SIZE),
              "spawn_positions": {name: list(map(float, value)) for name, value in positions.items()}}
    episode = env.run_trace(trace, frames_total=FRAMES_TOTAL, prelude_frames=PRELUDE_FRAMES,
                            skill="task_suite", params=params,
                            block=last["object"], destination=last["destination"])
    outcomes = subtask_outcomes(episode.tracks, subtasks)
    return {"episode": episode, "subtasks": subtasks, "outcomes": outcomes,
            "outcome": episode_outcome(outcomes), "family": family}


def phase_frames(records) -> dict:
    """Median executed duration of each phase, so a plan can be compiled without executing."""
    spans: dict[str, list[int]] = {}
    for record in records:
        if record["split"] != "train":
            continue
        for entry in record["skill"]["trace"]:
            spans.setdefault(entry["phase"], []).append(entry["frames"][1] - entry["frames"][0] + 1)
    return {phase: float(np.median(lengths)) for phase, lengths in spans.items()}


def split_for(index: int) -> str:
    return "train" if index % 20 < 14 else ("val" if index % 20 < 17 else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0,
                    help="episode index to start numbering at; shards write disjoint ranges")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--families", default=",".join(FAMILIES))
    args = ap.parse_args()

    families = [name.strip() for name in args.families.split(",") if name.strip()]
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise SystemExit(f"unknown families {sorted(unknown)}; expected {FAMILIES}")

    clips = args.out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    env = suite_env(width=args.size, height=args.size, fps=args.fps, seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    home = env.home_waypoint()
    records = []
    records_path = args.out / f"records_{args.start:06d}.jsonl"

    def flush():
        with records_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    for i in range(args.episodes):
        index = args.start + i
        # Offset by the 20-block so the family cycle does not alias with `split_for`'s. A plain
        # `index % 4` against a period-20 split means 4 divides 20 and every split position
        # carries a fixed family: `stack` then never appears in val at all. `validate_suite`
        # reassigns splits stratified by family anyway, but a shard should not be misleading on
        # its own.
        family = families[(index + index // 20) % len(families)]
        for _ in range(25):
            try:
                result = run_episode(env, rng, family)
                break
            except RuntimeError as error:
                if "IK failed" not in str(error) and "separated block positions" not in str(error):
                    raise
        else:
            raise RuntimeError(f"could not sample a reachable episode for index {index}")

        episode = result["episode"]
        name = f"ur5e_{index:05d}.mp4"
        iio.imwrite(clips / name, episode.frames, fps=args.fps, codec="libx264")
        records.append({
            "schema_version": SCHEMA_VERSION, "episode_id": f"ur5e_{index:05d}",
            "seed": args.seed, "split": split_for(index), "family": family,
            "observation": {"camera": "demo", "frames_path": f"clips/{name}",
                            "width": args.size, "height": args.size, "fps": args.fps,
                            "frames_total": len(episode.frames), "prelude_frames": PRELUDE_FRAMES,
                            "window_frames": WINDOW_FRAMES, "frame_times_s": episode.frame_times},
            "skill": {"name": episode.skill, "params": episode.params, "trace": episode.skill_trace},
            "tracks": episode.tracks,
            "state_before": episode.state_before, "state_after": episode.state_after,
            "subtask_outcomes": result["outcomes"],
            "outcome": result["outcome"],
        })
        # Written as we go: an hour of rendering should not be lost to a crash in the last minute.
        if (i + 1) % 25 == 0 or i + 1 == args.episodes:
            flush()
            solved = sum(r["outcome"]["success"] for r in records)
            print(f"{i + 1}/{args.episodes} [{family}] success {solved}/{len(records)}", flush=True)

    flush()
    (args.out / f"manifest_{args.start:06d}.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "episodes": len(records), "fps": args.fps,
        "frames_total": FRAMES_TOTAL, "prelude_frames": PRELUDE_FRAMES,
        "window_frames": WINDOW_FRAMES, "windows": WINDOWS,
        "block_spawn_low": list(SPAWN_LOW), "block_spawn_high": list(SPAWN_HIGH),
        "commanded_grasp_yaw": True, "oriented_blocks": True,
        "block_size": list(SUITE_BLOCK_SIZE), "families": families,
        "home_waypoint": home, "phase_frames": phase_frames(records),
        "block_names": list(scene.BLOCK_NAMES),
        "destinations": ["green_pad", *scene.BLOCK_NAMES]}, indent=2))
    print(f"wrote {len(records)} episodes to {records_path}")


if __name__ == "__main__":
    main()
