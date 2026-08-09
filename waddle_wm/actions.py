"""Skill-trace action encoding shared by the simulator, the trainer, and the verifier.

Every element of an action vector is recoverable from a skill trace alone, so the
same encoding serves executed episodes and plan-time compilation.
"""
from __future__ import annotations

import numpy as np

PHASES = ("idle", "approach", "descend", "close", "lift", "move", "place", "open", "retreat")
PHASE_ID = {name: i for i, name in enumerate(PHASES)}
GRIPPER_VALUES = {"close": 1.0, "open": 0.0}
ACTION_DIM = len(PHASES) + 4  # phase one-hot, waypoint xyz, gripper


def encode(phase, waypoint, gripper) -> np.ndarray:
    """Per-frame tracks -> (T, ACTION_DIM) action array."""
    phase, waypoint, gripper = np.asarray(phase, int), np.asarray(waypoint, float), np.asarray(gripper, float)
    onehot = np.zeros((len(phase), len(PHASES)))
    onehot[np.arange(len(phase)), phase] = 1.0
    return np.concatenate([onehot, waypoint, gripper[:, None]], axis=1).astype(np.float32)


def compile_plan(trace, phase_frames, home_waypoint, frames_total, prelude_frames) -> np.ndarray:
    """Skill trace -> (frames_total, ACTION_DIM), using median per-phase durations.

    This is the plan-time path: no execution, so phase durations come from
    `manifest.json -> phase_frames` instead of from a rendered episode.
    """
    phase, waypoint, gripper = [], [], []
    point, grip = np.asarray(home_waypoint, float), 0.0
    def emit(name, count):
        phase.extend([PHASE_ID[name]] * count); waypoint.extend([point] * count); gripper.extend([grip] * count)

    emit("idle", prelude_frames)
    for entry in trace:
        name = entry["phase"]
        if "target" in entry:
            point = np.asarray(entry["target"], float)
        grip = GRIPPER_VALUES.get(name, grip)
        emit(name, int(round(phase_frames[name])))
    emit("idle", max(0, frames_total - len(phase)))
    return encode(phase[:frames_total], waypoint[:frames_total], gripper[:frames_total])


def chunks(actions: np.ndarray, window_frames: int) -> np.ndarray:
    """(T, ACTION_DIM) -> (T // window_frames, window_frames, ACTION_DIM)."""
    count = len(actions) // window_frames
    return actions[:count * window_frames].reshape(count, window_frames, -1)
