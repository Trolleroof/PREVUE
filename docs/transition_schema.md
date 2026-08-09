# Transition Data Schema

One JSON record describes one MuJoCo skill execution. The rendered clip is stored separately as an `.mp4` or `.npz` file; the JSON record stores its relative path and all ground-truth labels.

```text
observation_before + skill trace -> observation_after + outcome
```

## Record

```json
{
  "schema_version": 1,
  "episode_id": "tabletop_000042",
  "seed": 42,
  "observation": {
    "camera": "frontal",
    "frames_path": "clips/tabletop_000042.npz",
    "frame_times_s": [0.1, 0.2, 0.3],
    "width": 256,
    "height": 256,
    "fps": 10
  },
  "state_before": {
    "block_pos": [0.0, 0.02, 0.021],
    "block_quat": [1.0, 0.0, 0.0, 0.0],
    "block_tilt_rad": 0.0,
    "gripper_pos": [0.0, 0.02, 0.22],
    "gripper_yaw": 0.0,
    "gripper_width": 0.07,
    "target_pos": [-0.22, 0.0],
    "grasped": false,
    "wall_clearance_y": 0.057,
    "target_distance": 0.221
  },
  "skill": {
    "name": "side_grasp",
    "params": {"side": 1.0, "standoff": 0.13, "grip_width": 0.05},
    "trace": [
      {"phase": "approach", "value": 0.15},
      {"phase": "descend", "value": 0.032},
      {"phase": "move_in", "value": 0.02},
      {"phase": "close", "value": 0.05},
      {"phase": "lift", "value": 0.24}
    ]
  },
  "state_after": {
    "block_pos": [0.0, 0.02, 0.021],
    "block_quat": [1.0, 0.0, 0.0, 0.0],
    "block_tilt_rad": 0.0,
    "gripper_pos": [0.0, 0.02, 0.24],
    "gripper_yaw": 1.571,
    "gripper_width": 0.05,
    "target_pos": [-0.22, 0.0],
    "grasped": false,
    "wall_clearance_y": 0.057,
    "target_distance": 0.221
  },
  "outcome": {
    "success": false,
    "failure_mode": "collision_with_bin_wall"
  }
}
```

## Contract

`state_before` and `state_after` use the exact dictionary returned by `TabletopEnv.state()`. All positions are in MuJoCo world coordinates in meters; quaternions use MuJoCo order `[w, x, y, z]`; angles are radians.

Allowed skills:

- `top_grasp`
- `side_grasp`
- `align`

Allowed trace phases:

- `approach`
- `descend`
- `move_in`
- `close`
- `lift`
- `move`
- `settle`

Failure modes:

- `collision_with_bin_wall`
- `slip`
- `grip_too_weak`
- `missed`
- `toppled`
- `no_contact`
- `overshoot`
- `undershoot`

## Model Inputs and Targets

The frozen V-JEPA backbone consumes the observation frames. The PyTorch probe consumes the V-JEPA embedding plus a numeric encoding of `skill.name`, `skill.params`, and `skill.trace`.

The probe predicts:

- `outcome.success`
- `outcome.failure_mode`
- selected values from `state_after`, starting with `block_pos`, `grasped`, and `target_distance`

The model must never train on `state_after` as an input. It is a future-state target.
