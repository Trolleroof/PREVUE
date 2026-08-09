# MuJoCo execution environment

The first execution environment for the Waddle skill-level world model: a
repeatable tabletop scene with ground-truth object poses, skill-level control,
and rendered camera observations for V-JEPA 2.

## Setup

MuJoCo is a project dependency, so nothing beyond the normal sync is needed:

```bash
uv sync
```

macOS offscreen rendering works through MuJoCo's native CGL renderer — no
`MUJOCO_GL` override, no X server, no ROS.

Verify:

```bash
uv run python -m waddle_wm.sim.test_env
```

## Scene

`assets/tabletop.xml` contains:

| element      | detail                                                              |
| ------------ | ------------------------------------------------------------------- |
| robot        | OpenArm v0.3, 7-DOF arm with a parallel gripper                     |
| blocks       | Red, blue, and yellow 70 mm cubes                                   |
| target       | Green circular landing zone                                          |
| camera       | `demo`                                                               |

The gripper is gravity-compensated so commanded positions are the positions it
actually reaches. Jaw width is the distance between the finger origins; the
inner faces sit 10 mm inside it, so the block needs > 62 mm to clear and a
command near 30 mm to be a firm squeeze.

## Skills

Skills are executed as position-setpoint segments, never motor torques.

| skill        | trace                                        | main parameters              |
| ------------ | -------------------------------------------- | ---------------------------- |
| `top_grasp`  | approach, descend, close, lift               | `offset_x/y`, `grip_width`   |
| `side_grasp` | approach, descend, move_in, close, lift      | `side`, `standoff`, `grip_width` |
| `align`      | approach, move, settle                       | `push_to_x`, `standoff`      |

`side_grasp` yaws the wrist 90° and comes in laterally along y, so `side=+1`
approaches straight across the bin wall and collides — this is the failure the
verifier is meant to predict. `align` pushes with narrowed jaws.

## Outcome labels

Grasps succeed when the block ends above 90 mm with both fingers in contact.
`align` succeeds when the block ends inside the target region and upright.
Failure modes: `collision_with_bin_wall`, `slip`, `grip_too_weak`, `missed`,
`toppled`, `overshoot`, `undershoot`, `no_contact`.

## Dataset

```bash
uv run python -m waddle_wm.sim.generate_dataset --episodes 200
```

Writes `data/mujoco_tabletop/` (gitignored; the generator is seeded, so it is
reproducible from the command above). Render the requested demo with:

```bash
uv run python -m waddle_wm.sim.render_openarm_demo
```

This writes `data/openarm_pick_place.mp4`.

- `clips/NNNN_<skill>.mp4` — 256×256 at 10 fps, roughly 3–5 s per episode
- `records.jsonl` — one record per episode

```json
{
  "id": "0000", "split": "train", "clip": "clips/0000_top_grasp.mp4",
  "n_frames": 33, "fps": 10, "camera": "frontal",
  "skill": "top_grasp", "params": {"grip_width": 0.03},
  "skill_trace": [["approach", 0.22], ["descend", 0.038], ["close", 0.03], ["lift", 0.2]],
  "state_before": {"block_pos": [0.0, 0.0, 0.021], "...": "..."},
  "state_after":  {"block_pos": [0.0, 0.0, 0.165], "...": "..."},
  "success": true, "failure_mode": null
}
```

Split is 70/15/15 by episode index. At 200 episodes the run is ~48% success
with all five common failure modes represented, so the outcome probe has a
non-degenerate majority-class baseline to beat.

Useful flags: `--episodes`, `--seed`, `--camera {frontal,oblique}`, `--size`,
`--fps`, `--out`.
