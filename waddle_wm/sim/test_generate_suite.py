"""Check the task suite's labelling and plan encoding on hand-built trajectories.

Scoring is what every downstream number rests on, so it is tested against tracks written by
hand rather than against the simulator: a physics change should not be able to silently
redefine what "stacked" means.

    uv run python -m waddle_wm.sim.test_generate_suite
"""
from __future__ import annotations

import math

import numpy as np

from waddle_wm import plan_encoding
from waddle_wm.sim import relling_scene as scene
from waddle_wm.sim.generate_suite import (BLOCK_HEIGHT, FRAMES_TOTAL, STACK_XY_TOL, SUITE_BLOCK_SIZE,
                                          build_trace, episode_outcome, plan_subtasks,
                                          subtask_outcomes, subtask_trace, wrap_yaw_deg)
from waddle_wm.sim.env import LIFT_THRESHOLD, TARGET_RADIUS
from waddle_wm.sim.validate_suite import misalignment_deg
from waddle_wm.train_task_suite_world_model import episode_features, trace_segments

NAMES = list(scene.BLOCK_NAMES)
PAD = np.array([0.50, 0.30])
REST_Z = SUITE_BLOCK_SIZE[2]
FAILURES: list[str] = []


def check(condition, message):
    if condition:
        print(f"PASS  {message}")
    else:
        print(f"FAIL  {message}")
        FAILURES.append(message)


def tracks_for(paths: dict[str, list[tuple]], frames: int = FRAMES_TOTAL) -> dict:
    """Per-frame `all_block_pos` from a per-block list of (frame, xyz) keyframes, held forward."""
    positions = np.zeros((frames, len(NAMES), 3))
    for index, name in enumerate(NAMES):
        current = np.array([0.4, -0.2, REST_Z])
        keyframes = dict(paths.get(name, []))
        for frame in range(frames):
            if frame in keyframes:
                current = np.asarray(keyframes[frame], dtype=float)
            positions[frame, index] = current
    return {"all_block_pos": positions.tolist()}


def subtask(source, destination, target_xy, place_z=None, grasp_yaw_deg=0.0, block_yaw_deg=0.0):
    return {"object": source, "destination": destination, "target_xy": list(target_xy),
            "goal_xy": list(PAD) if destination == "green_pad" else None,
            "place_z": place_z if place_z is not None else 0.015,
            "grasp_offset_xy": [0.0, 0.0], "block_yaw_deg": block_yaw_deg,
            "grasp_yaw_deg": grasp_yaw_deg, "yaw_aligned": grasp_yaw_deg == block_yaw_deg,
            "planted_grasp_fault": False, "planted_target_fault": False}


def main():
    # --- a pad placement succeeds only if the block was lifted and ended on the pad -----------
    lifted_to_pad = tracks_for({"red_block": [(10, [0.4, -0.2, 0.20]), (30, [*PAD, REST_Z])]})
    task = subtask("red_block", "green_pad", PAD)
    outcome = subtask_outcomes(lifted_to_pad, [task])[0]
    check(outcome["success"] and outcome["lifted"] and outcome["placed"],
          "lifted and placed on the pad is a success")

    dragged = tracks_for({"red_block": [(30, [*PAD, REST_Z])]})     # never left the table
    outcome = subtask_outcomes(dragged, [task])[0]
    check(not outcome["success"] and not outcome["lifted"] and outcome["failure_mode"] == "missed",
          "a block that reached the pad without ever being lifted is `missed`, not a success")

    off_pad = tracks_for({"red_block": [(10, [0.4, -0.2, 0.20]),
                                        (30, [PAD[0] + TARGET_RADIUS + 0.02, PAD[1], REST_Z])]})
    outcome = subtask_outcomes(off_pad, [task])[0]
    check(not outcome["success"] and outcome["failure_mode"] == "target_miss",
          "a block lifted and dropped off the pad is a `target_miss`")

    # The goal is the pad, not the plan's own aim point: a plan that aims wide has missed.
    wide = subtask("red_block", "green_pad", PAD + np.array([0.30, 0.0]))
    landed_where_aimed = tracks_for({"red_block": [(10, [0.4, -0.2, 0.20]),
                                                   (30, [PAD[0] + 0.30, PAD[1], REST_Z])]})
    outcome = subtask_outcomes(landed_where_aimed, [wide])[0]
    check(not outcome["success"],
          "a plan that aims off the pad fails even when the block lands exactly where aimed")

    # --- stacking is judged on the support's footprint and height ------------------------------
    support_xy = [0.45, 0.25]
    stack_task = subtask("blue_block", "red_block", support_xy, place_z=REST_Z + BLOCK_HEIGHT)
    stacked = tracks_for({"red_block": [(0, [*support_xy, REST_Z])],
                          "blue_block": [(10, [0.4, -0.2, 0.20]),
                                         (30, [*support_xy, REST_Z + BLOCK_HEIGHT])]})
    outcome = subtask_outcomes(stacked, [stack_task])[0]
    check(outcome["success"] and outcome["resting"], "a block resting on the support is stacked")

    beside = tracks_for({"red_block": [(0, [*support_xy, REST_Z])],
                         "blue_block": [(10, [0.4, -0.2, 0.20]),
                                        (30, [support_xy[0], support_xy[1], REST_Z])]})
    outcome = subtask_outcomes(beside, [stack_task])[0]
    check(not outcome["success"] and outcome["failure_mode"] == "not_stacked",
          "a block at the support's xy but still on the table is `not_stacked`")

    far = tracks_for({"red_block": [(0, [*support_xy, REST_Z])],
                      "blue_block": [(10, [0.4, -0.2, 0.20]),
                                     (30, [support_xy[0] + STACK_XY_TOL + 0.01, support_xy[1],
                                           REST_Z + BLOCK_HEIGHT])]})
    outcome = subtask_outcomes(far, [stack_task])[0]
    check(not outcome["success"], "a block past the support's footprint is not stacked")

    # The support moves: the goal follows it, because the goal is the block, not a coordinate.
    moved = tracks_for({"red_block": [(20, [0.60, 0.10, REST_Z])],
                        "blue_block": [(10, [0.4, -0.2, 0.20]),
                                       (30, [0.60, 0.10, REST_Z + BLOCK_HEIGHT])]})
    outcome = subtask_outcomes(moved, [stack_task])[0]
    check(outcome["success"], "stacking is scored against where the support actually ended up")

    # --- an episode succeeds only when every subtask does ---------------------------------------
    both = [{"success": True}, {"success": True}]
    check(episode_outcome(both)["success"], "both subtasks succeeding is an episode success")
    mixed = [{"success": True, "failure_mode": None}, {"success": False, "failure_mode": "missed"}]
    result = episode_outcome(mixed)
    check(not result["success"] and result["failed_subtask"] == 1 and result["failure_mode"] == "missed",
          "one failed subtask fails the episode and names which one")
    first_bad = [{"success": False, "failure_mode": "target_miss"}, {"success": True, "failure_mode": None}]
    check(episode_outcome(first_bad)["failed_subtask"] == 0,
          "the reported failure is the first subtask that broke")

    # --- headings ---------------------------------------------------------------------------
    check(abs(wrap_yaw_deg(120.0) - -60.0) < 1e-9 and abs(wrap_yaw_deg(-100.0) - 80.0) < 1e-9,
          "headings fold into +-90 degrees, since the jaws are symmetric")
    check(abs(misalignment_deg({"grasp_yaw_deg": 85.0, "block_yaw_deg": -85.0}) - 10.0) < 1e-9,
          "misalignment wraps: 85 and -85 degrees are 10 degrees apart, not 170")

    # --- the trace carries the heading, and splits back into subtasks -------------------------
    task_a = subtask("red_block", "green_pad", PAD, grasp_yaw_deg=30.0)
    task_b = subtask("blue_block", "red_block", PAD, place_z=REST_Z + BLOCK_HEIGHT, grasp_yaw_deg=-40.0)
    trace = build_trace([task_a, task_b], {"red_block": [0.35, -0.2, REST_Z],
                                           "blue_block": [0.55, -0.2, REST_Z],
                                           "yellow_block": [0.65, -0.2, REST_Z]})
    segments = trace_segments(trace)
    check(len(segments) == 2 and all(len(s) == 8 for s in segments),
          "a two-subtask trace splits back into two eight-phase segments")
    grasp_yaw, approach_yaw = plan_encoding.trace_yaws(segments[1])
    check(grasp_yaw is not None and abs(math.degrees(grasp_yaw) - -40.0) < 1e-6,
          "the second segment carries the second subtask's commanded heading")

    single = subtask_trace(task_a, [0.35, -0.2])
    check(all(entry.get("yaw") is not None for entry in single
              if entry["phase"] in ("approach", "descend", "lift")),
          "the heading is pinned through approach, descent and lift")

    # --- two candidates differing only in heading must not encode identically ------------------
    record = {"episode_id": "t", "family": "ordered_stack",
              "skill": {"trace": trace, "params": {"subtasks": [task_a, task_b]}}}
    initial = np.array([0.35, -0.2, REST_Z, 0.55, -0.2, REST_Z, 0.65, -0.2, REST_Z, 0.0, 0.0, 0.3])
    plans, tasks, mask = episode_features(record, initial, tuple(NAMES), 2)
    turned = subtask("red_block", "green_pad", PAD, grasp_yaw_deg=-30.0)
    other_trace = build_trace([turned, task_b], {"red_block": [0.35, -0.2, REST_Z],
                                                 "blue_block": [0.55, -0.2, REST_Z],
                                                 "yellow_block": [0.65, -0.2, REST_Z]})
    other = {"episode_id": "t2", "family": "ordered_stack",
             "skill": {"trace": other_trace, "params": {"subtasks": [turned, task_b]}}}
    other_plans, _, _ = episode_features(other, initial, tuple(NAMES), 2)
    check(not np.allclose(plans, other_plans),
          "two plans differing only in commanded heading get different model inputs")
    check(mask.tolist() == [1.0, 1.0], "both subtask slots are marked present")

    # A one-subtask family leaves the second slot empty rather than repeating the first.
    solo = {"episode_id": "t3", "family": "pad_place",
            "skill": {"trace": subtask_trace(task_a, [0.35, -0.2]), "params": {"subtasks": [task_a]}}}
    _, _, solo_mask = episode_features(solo, initial, tuple(NAMES), 2)
    check(solo_mask.tolist() == [1.0, 0.0], "a one-subtask family masks its second slot")

    # --- the second subtask aims at where the plan will have left the support ------------------
    rng = np.random.default_rng(0)
    positions = {"red_block": [0.35, -0.2, REST_Z], "blue_block": [0.55, -0.2, REST_Z],
                 "yellow_block": [0.65, -0.2, REST_Z]}
    yaws = {name: 0.0 for name in NAMES}
    planned = plan_subtasks(rng, "ordered_stack", positions, yaws, PAD)
    support = planned[1]["destination"]
    check(support == planned[0]["object"],
          "ordered_stack stacks the second block onto the one the first subtask placed")
    check(np.linalg.norm(np.asarray(planned[1]["target_xy"]) -
                         np.asarray(positions[support][:2])) > 0.1,
          "and aims at the pad, not at the support's spawn position it has since left")

    print(f"\n{'FAILED' if FAILURES else 'OK'}: {len(FAILURES)} failing checks")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
