"""Turn schema-3 episode records into latent-transition examples.

One example is (context window k, action chunk over window k+1) -> (target window
k+1). See docs/transition_schema.md; nothing here touches pixels or the encoder.
"""
from __future__ import annotations

import numpy as np

from waddle_wm.actions import chunks, compile_plan, encode
from waddle_wm.sim.env import LIFT_THRESHOLD, TARGET_RADIUS

STATE_KEYS = ("block_x", "block_y", "block_z", "pinch_x", "pinch_y", "pinch_z")
BINARY_KEYS = ("lifted", "in_target")


def anchors(manifest) -> np.ndarray:
    """Last frame index of each window, e.g. 7, 15, 23, 31, 39, 47."""
    window = manifest["window_frames"]
    return np.arange(manifest["windows"]) * window + window - 1


def executed_actions(record) -> np.ndarray:
    """(frames_total, ACTION_DIM) from what the simulator actually commanded."""
    tracks = record["tracks"]
    return encode(tracks["phase"], tracks["waypoint"], tracks["gripper"])


def planned_actions(record, manifest) -> np.ndarray:
    """(frames_total, ACTION_DIM) compiled from the skill trace, as at plan time."""
    return compile_plan(record["skill"]["trace"], manifest["phase_frames"], manifest["home_waypoint"],
                        manifest["frames_total"], manifest["prelude_frames"])


def window_states(record, manifest) -> np.ndarray:
    """(windows, 8) grounded state at each anchor: block xyz, pinch xyz, lifted, in_target."""
    tracks, index = record["tracks"], anchors(manifest)
    block, pinch = np.asarray(tracks["block_pos"])[index], np.asarray(tracks["pinch_pos"])[index]
    lifted = (np.asarray(tracks["max_block_z"])[index] > LIFT_THRESHOLD).astype(float)
    in_target = (np.asarray(tracks["target_distance"])[index] <= TARGET_RADIUS).astype(float)
    return np.concatenate([block, pinch, lifted[:, None], in_target[:, None]], axis=1).astype(np.float32)


def build(records, manifest, planned=False) -> dict:
    """Stack every episode's transitions into flat arrays."""
    steps = manifest["windows"] - 1
    action = np.stack([chunks(planned_actions(r, manifest) if planned else executed_actions(r), manifest["window_frames"])[1:]
                       for r in records])                                   # (episodes, steps, window, action)
    state = np.stack([window_states(r, manifest) for r in records])         # (episodes, windows, 8)
    return {
        "episode_ids": [r["episode_id"] for r in records],
        "splits": np.array([r["split"] for r in records]),
        "success": np.array([float(r["outcome"]["success"]) for r in records], dtype=np.float32),
        "failure_mode": np.array([r["outcome"]["failure_mode"] or "none" for r in records]),
        "action": action.astype(np.float32),
        "state": state,
        "steps": steps,
    }


def flatten(data, key):
    """(episodes, steps, ...) -> (episodes * steps, ...) with matching episode index."""
    episodes, steps = data[key].shape[:2]
    return data[key].reshape(episodes * steps, *data[key].shape[2:]), np.repeat(np.arange(episodes), steps)
