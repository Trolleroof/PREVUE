# MuJoCo execution environment

The first execution environment for the Waddle skill-level world model: a
physical UR5e + Robotiq 2F-85 pick-and-place scene with rendered camera clips
and ground-truth future state/outcome labels for V-JEPA 2.

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

`relling_scene.py` assembles the UR5e, Robotiq gripper, table, three blocks,
and landing zone from `assets/ur5e.xml` and `assets/2f85.xml`.

| element      | detail                                                              |
| ------------ | ------------------------------------------------------------------- |
| robot        | UR5e, 6-DOF arm with a Robotiq 2F-85 gripper                        |
| blocks       | Red, blue, and yellow 36 mm cubes                                   |
| target       | Green circular landing zone                                          |
| camera       | `demo`                                                               |

The Robotiq actuator is commanded in its native 0–255 range; its contact
physics, rather than a weld or kinematic attachment, determines whether a
block is actually lifted.

## Skill

`pick_place` executes `approach → descend → close → lift → move → place → open
→ retreat` through UR5e inverse kinematics and physical Robotiq contacts.
The generator deliberately asks for some targets outside the landing zone to
produce useful `target_miss` negatives.

## Outcome labels

Episodes succeed when the block is physically lifted above 90 mm and ends in
the target region. Failure modes are `missed` and `target_miss`.

## Dataset

```bash
uv run python -m waddle_wm.sim.generate_dataset --episodes 200
```

Writes `data/ur5e_tabletop/` (gitignored; the generator is seeded, so it is
reproducible from the command above). Render the requested demo with:

```bash
uv run python -m waddle_wm.sim.render_ur5e_demo
```

This writes `data/ur5e_pick_place.mp4`.

- `clips/ur5e_NNNN.mp4` — 256×256 at 10 fps
- `records.jsonl` — one record per episode

```json
{
  "episode_id": "ur5e_0000", "split": "train",
  "observation": {"frames_path": "clips/ur5e_0000.mp4", "fps": 10},
  "skill": {"name": "pick_place", "params": {"target_xy": [0.5, 0.3]}},
  "outcome": {"success": true, "failure_mode": null}
}
```

Split is 70/15/15 by episode index. The seeded generator mixes reachable
landing-zone targets with reachable misses for a non-degenerate outcome probe.

Useful flags: `--episodes`, `--seed`, `--size`, `--fps`, `--out`,
`--block-spawn-low`, `--block-spawn-high`, `--append`.

## Validation

```bash
uv run python -m waddle_wm.sim.validate_dataset --data data/ur5e_wm_wide
```

Checks schema version, frame grid, per-frame track lengths, spawn-box
conformance, split sizes, outcome balance, duplicate ids, duplicate scenes,
cross-split scene leakage, and decodes `--render-sample` clips (50 by default;
`0` skips). Exits non-zero on any failure.

Because `--append` keeps whatever distribution the earlier episodes had, the
spawn check compares the first and second half of the corpus: a half that spans
a visibly narrower box means episodes were appended to an older corpus rather
than regenerated. `docs/results.md` §0 records the output for the 5000-episode
wide corpus. The validator has its own check:

```bash
uv run python -m waddle_wm.sim.test_validate_dataset
```
