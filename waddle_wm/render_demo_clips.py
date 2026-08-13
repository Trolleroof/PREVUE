"""Render the plan each recorded demo arm actually ran, as MuJoCo footage for the demo page.

    uv run python -m waddle_wm.render_demo_clips

For every trace in `results/demo`, this runs *the waypoints stored in that trace* — the flawed
opening plan for the `none` arm, the approved repaired plan for the verified arms — on the same
scene, and writes `{scenario}.{arm}.clip.mp4`. No Claude call, no verifier, no checkpoint: the
decision was already made and recorded, so this only re-executes it for the camera.

The printed outcome per clip is checked against the outcome in the trace; a mismatch means the
footage is not showing what the trace says happened, and the run fails rather than writing a
misleading clip.

`--width/--height` set the render size (the training pipeline uses 256; the page wants larger).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from waddle_wm.sim.env import FRAMES_TOTAL, PRELUDE_FRAMES, WINDOW_FRAMES, TabletopEnv

REPO = Path(__file__).resolve().parent.parent
ARMS = ("none", "rules", "world-model")
SCENARIOS = ("grasp_miss", "place_miss")


def executed_trace(record: dict) -> tuple[list[dict], int]:
    """The waypoints the arm ran, and which round they came from.

    `none` never verifies, so it runs round 0. A verified arm runs the last round it recorded,
    which is the plan its verifier approved.
    """
    index = 0 if record["arm"] == "none" else len(record["rounds"]) - 1
    plan = record["rounds"][index]["plan"]
    steps = plan.get("steps") or []
    return (plan.get("trace") or steps[0]["trace"]), index


def render(record: dict, width: int, height: int, seed: int) -> tuple[np.ndarray, dict]:
    """Run one recorded plan on its recorded scene and hand back the frames and the outcome."""
    env = TabletopEnv(seed=seed, width=width, height=height)
    env.reset(list(record["block_xy"]))
    env.observation_frames(WINDOW_FRAMES)        # the same pre-execution window the agent renders
    trace, _ = executed_trace(record)
    episode = env.run_trace(trace, frames_total=FRAMES_TOTAL, prelude_frames=PRELUDE_FRAMES,
                            block="red_block", destination="green_pad")
    state = episode.state_after
    return episode.frames, {
        "success": episode.success,
        "failure_mode": episode.failure_mode,
        "final_block_xy": [round(float(v), 4) for v in state["block_pos"][:2]],
        "target_distance": round(float(state["target_distance"]), 4),
        "max_block_z": round(float(state["max_block_z"]), 4),
    }


def agrees(fresh: dict, record: dict, tolerance: float = 0.03) -> bool:
    """Does the footage show the outcome the trace recorded? Positions may wobble; verdicts may not."""
    if fresh["success"] is not record["executed_success"]:
        return False
    if fresh["failure_mode"] != record["failure_mode"]:
        return False
    return abs(fresh["target_distance"] - record["target_distance"]) <= tolerance


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces", type=Path, default=REPO / "results" / "demo")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    disagreed = []
    for scenario in SCENARIOS:
        for arm in ARMS:
            record = json.loads((args.traces / f"{scenario}.{arm}.json").read_text())
            frames, outcome = render(record, args.width, args.height, args.seed)
            out = args.traces / f"{scenario}.{arm}.clip.mp4"
            iio.imwrite(out, np.asarray(frames), fps=args.fps, codec="libx264")
            ok = agrees(outcome, record)
            disagreed += [] if ok else [f"{scenario}/{arm}"]
            print(f"{out.name}: {len(frames)} frames · "
                  f"{'success' if outcome['success'] else 'failure ' + str(outcome['failure_mode'])} · "
                  f"{outcome['target_distance']} m from the pad · "
                  f"{'matches trace' if ok else 'DISAGREES WITH TRACE'}")

    if disagreed:
        raise SystemExit(f"footage does not match the recorded outcome for: {', '.join(disagreed)}")


if __name__ == "__main__":
    main()
