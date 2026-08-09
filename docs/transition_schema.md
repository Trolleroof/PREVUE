# UR5e Transition Data Schema

One JSONL record describes one physical UR5e/Robotiq `pick_place` execution.

```text
observation frames + skill trace -> future state + outcome
```

```json
{
  "schema_version": 2,
  "episode_id": "ur5e_000042",
  "seed": 42,
  "split": "train",
  "observation": {
    "camera": "demo", "frames_path": "clips/ur5e_0042.mp4",
    "frame_times_s": [0.1, 0.2], "width": 256, "height": 256, "fps": 10
  },
  "skill": {
    "name": "pick_place", "params": {"target_xy": [0.5, 0.3]},
    "trace": [{"phase": "approach", "target": [0.38, -0.18, 0.24]}]
  },
  "state_before": {"block_pos": [0.38, -0.18, 0.018], "...": "..."},
  "state_after": {"block_pos": [0.5, 0.3, 0.018], "...": "..."},
  "outcome": {"success": true, "failure_mode": null}
}
```

`state_before` and `state_after` contain the red block pose, pinch-site position,
target position, target distance, and maximum observed lift. Positions are meters
in MuJoCo world coordinates and quaternions are `[w, x, y, z]`.

The frozen V-JEPA backbone consumes `observation` frames. The probe consumes its
embedding plus `skill`; `state_after` and `outcome` are targets only. A plan with a
target outside the green landing zone is labeled `target_miss`; a block that never
lifts above 90 mm is `missed`.
