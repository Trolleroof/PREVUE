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
| table        | plane at `z = 0`                                                     |
| block        | 42 mm red cube, free joint, 60 g                                     |
| bin wall     | static box at `y = +0.085`, 100 mm tall — the obstacle               |
| target       | 45 mm radius region at `(-0.22, 0)` for `align`                      |
| gripper      | 3 slide joints (x, y, z), a wrist yaw hinge, and two mirrored jaws   |
| cameras      | `frontal` and `oblique`                                              |

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
reproducible from the command above):

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

---

# YAM arm execution environment

A second execution environment that runs the same three skills on a real robot
model: the [I2RT YAM](https://i2rt.com/products/yam-manipulator) 6-DoF
manipulator from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`i2rt_yam`, MIT). The pseudo-gripper's three slide joints are replaced by an
actual arm, so skills become Cartesian waypoints solved by inverse kinematics
and tracked by the arm's position actuators.

```bash
uv run python -m waddle_wm.sim.test_yam_env
```

- `assets/yam/` — the vendored model, byte-identical to upstream
- `assets/tabletop_yam.xml` — the scene; it `<attach>`es the arm rather than
  including it, which is why the vendored file needs no edits
- `sim/yam_env.py` — `YamTabletopEnv`, same API and `Episode` type as `TabletopEnv`

Skill parameters, outcome labels, and the coordinate frame are unchanged: poses
are workspace-local, with the block at the origin, so records from the two
environments line up. The workspace sits at world `(0.50, 0, 0)`; the arm is
bolted to the table at the origin.

## Inverse kinematics

Position is the primary task and orientation is solved in its nullspace. That
split is not a nicety: sampling 200k random configurations found that poses
holding an *exactly* vertical approach near the table are a very thin sliver of
this arm's configuration space (8 in 120k in the forward region), while the
positions themselves are always reachable. Demanding both exactly makes a
solver trade away reach; solving position first and letting the wrist tilt a few
degrees hits every skill waypoint to under a millimetre.

Three things were needed to make it reliable, each fixing a measured failure:

| problem | fix |
| ------- | --- |
| random restarts almost never land in the down-pointing basin | seed from a library of 40k sampled forward-kinematics poses, nearest first |
| a joint pinned on its limit absorbs the step and the descent stalls | drop pinned joints from the Jacobian |
| consecutive waypoints pick elbow-flipped solutions, sweeping the arm through the scene | among solutions that reach the target, prefer the one nearest the current configuration |

Waypoints are interpolated along a straight Cartesian line and each segment ends
with a short settle, because the position servos lag by about a centimetre while
a segment is still ramping.

## What the real gripper changes

Measured off the model, and the reason each scene value differs from the
pseudo-gripper scene:

| measurement | value | consequence |
| ----------- | ----- | ----------- |
| TCP (centre of the gripping plates) | `site + (0.0305, 0, -0.0247)` in the site frame | the `grasp_site` itself is neither between nor level with the plates |
| plate reach below the TCP, top-down | 19.5 mm | a top grasp holds the block's top ~16 mm, so `GRASP_Z` is 45 mm |
| wrist reach below the TCP | 18.4 mm | the fingers protrude only ~1 mm past the wrist housing |
| gripper half-width, closed | 55 mm | wider than the old wall clearance |
| jaw gap vs command | gap ≈ command + 0.8 mm | 0.030 only kisses a 42 mm block and it squirts out on the lift; the default grip is 0.024 |

Two scene values therefore differ from `tabletop.xml`, both forced by the real
hardware and both marked in the XML:

- the bin wall moved out (inner face `y = 0.107`, was `0.077`) — at the old
  position YAM's knuckles clipped it during an ordinary top grasp
- the tabletop friction dropped to 0.4 (was 1.5) — a 42 mm cube tips rather than
  slides once `mu` exceeds about half-width over contact height

## Status

| skill | result |
| ----- | ------ |
| `top_grasp` | works, 5/5 on randomised scenes |
| `side_grasp` (`side=-1`) | works, 5/5; approaches horizontally, which is what the motion looks like on a real arm rather than a top-down wrist roll |
| `side_grasp` (`side=+1`) | fails as intended with `collision_with_bin_wall` |
| `align` | **does not work on this arm** |

`align` is a genuine hardware limitation, not a tuning gap. Pushing the block
needs contact below its centre of mass, and this gripper cannot deliver it:

- top-down, the plate bottom (TCP − 19.5 mm) and the wrist bottom (TCP − 18.4 mm)
  are within a millimetre of each other, so any height where the plates touch a
  42 mm block also buries the wrist in it, and any height that clears the wrist
  clears the block entirely
- horizontally, with the fingers level at the block's mid-height, the forearm
  goes through the table and the arm cannot hold the pose
- angled 45° down, the entry pose is reachable on paper but the arm does not
  track it cleanly

A wrist yaw is also limited: YAM can only roll about ±70° around a vertical
approach before joint 5 and joint 6 saturate, so the original 90°-yawed top-down
side grasp is impossible and is done as a horizontal approach instead.
