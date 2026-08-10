# The LLM planner and the repair loop

This is the layer [issue #17](https://github.com/Trolleroof/skill-level-world-model/issues/17)
asked for: Claude Opus 5 proposes a structured skill trace, the local world model
imagines it, and Claude gets one repair at a time until the plan is approved,
abandoned, or run.

```text
natural-language command          camera
        |                            |
        |                  bounding_box(text -> box)
        |                  detect_in_base(box -> point)
        |                            |
        v                            v
  Claude Opus 5  <─── detections ────┘
        |
        └──────────────>  structured skill trace (waypoint program)
        ^                                   |
        |  one repair at a time             v
        |                        frozen V-JEPA encodes the 8-frame
        |                        pre-execution window
        |                                   |
        |                                   v
        └────── imagined outcome <── action-conditioned latent rollout
                (p(success), p(lifted), p(in zone),         |
                 uncertainty, likely failure)               v
                                    approved -> damped least-squares IK -> MuJoCo
```

Objects are located by the camera, not by reading simulator state: a text query becomes a
pixel box, and the depth buffer turns that box into a point in the frame the arm is
commanded in. That perception layer is modelled on the primitives
[Waddle Labs publishes](https://www.waddlelabs.ai/research/introducing-waddle).

Three commands, three entry points:

```bash
uv run python -m waddle_wm.server
```

```bash
uv run python -m waddle_wm.agent --instruction "put the red block on the green pad"
```

```bash
uv run python -m waddle_wm.test_agent --live 30
```

The server is the browser demo: the episode on the left, a chat bar on the right, each
step of the loop streamed into the transcript as it happens. The viewport is empty until
you prompt it; afterwards it plays the recorded episode at a true 10 fps.
`TabletopEnv.on_frame` fires as each frame is captured inside the physics loop, which the
server uses to shadow the run at 720 px — the page never plays the 256 px frames the
verifier and the dataset need. The agent CLI is
the same loop headless, writing `results/agent/<timestamp>.json` and `.mp4`. The
test script checks the plan contract offline and, with `--live`, measures the
verifier against physics on freshly rendered scenes.

## Claude runs through the CLI, not an API key

`waddle_wm.planner` shells out to `claude -p ... --output-format json`, so the
planner uses the Claude Code login already on the machine. One stateless turn per
call, with the plan schema as `--system-prompt` and every tool switched off:

```python
[binary, "-p", prompt, "--output-format", "json", "--model", model,
 "--system-prompt", SYSTEM_PROMPT, "--allowed-tools", "", "--strict-mcp-config",
 "--disable-slash-commands", "--max-turns", "1"]
```

Cost and session id come back in the envelope and land in the run log — a propose /
verify / repair / execute cycle is about **$0.09–$0.13 and 10–25 s** on Opus 5.
`--model claude-haiku-4-5-20251001` works too and is roughly 20x cheaper.

## The plan format

Claude answers with one JSON object and nothing else:

```json
{"intent": "one line restating the command",
 "action": "execute",
 "trace": [{"phase": "approach", "target": [0.44, -0.21, 0.24]},
           {"phase": "descend",  "target": [0.44, -0.21, 0.015]},
           {"phase": "close"},
           {"phase": "lift",     "target": [0.44, -0.21, 0.24]},
           {"phase": "move",     "target": [0.43, 0.30, 0.30]},
           {"phase": "place",    "target": [0.43, 0.30, 0.015]},
           {"phase": "open"},
           {"phase": "retreat",  "target": [0.43, 0.30, 0.24]}],
 "note": "what I aimed at, or what I changed and why"}
```

That is exactly the format `actions.compile_plan` compiles and `TabletopEnv.run_trace`
executes, so one object is both the thing scored and the thing run — no translation
layer that could drift. `validate()` rejects unknown phases, missing targets,
out-of-workspace waypoints and over-long traces, and the rejection text is handed
straight back to Claude for another attempt (`planner.plan` retries twice).

`action: "abstain"` with an empty trace is a first-class answer, used when the
command names something that is not in the observation or when the operator's
constraints contradict each other.

## Where the 3D positions come from

Modelled on the primitive layer [Waddle Labs
describes](https://www.waddlelabs.ai/research/introducing-waddle) — `bounding_box`
(text query → box in frame), `detect_in_base` (box → point in the robot base frame),
`approach_until` (waypoints + stop criterion → trajectory), `reset_home` — with skills
composed on top of primitives and a program composed of skills. `waddle_wm/perception.py`
implements the first three, and the planner reads nothing else:

| primitive | here |
| --- | --- |
| `bounding_box("the red block")` | MuJoCo's segmentation buffer stands in for an open-vocabulary detector: a word list resolves the phrase to one object, the mask gives the tightest pixel box. This is the one piece that is not real perception — swapping in a detector means replacing this method and nothing else. |
| `detect_in_base(box)` | Real geometry. Every pixel in the box nearer than the background is unprojected through the intrinsics (fx = fy = ½·h / tan(½·fovy), fovy 45°) and the camera extrinsics into a point cloud, then the cloud's centroid is pushed back along the view axis by half its depth extent to recover the object centre rather than its front face. |
| `approach_until(waypoints, stop)` | `TabletopEnv.approach_until`, checked on every physics step because the arm lags its command. The pick-and-place path deliberately does not use it: an early stop would desynchronise phase durations from the `phase_frames` the verifier compiles against. |
| `reset_home()` | `TabletopEnv.reset`. |

The base frame is the frame waypoints are commanded in, which here is MuJoCo's world
frame. The `base` body's own frame is rotated 180° about z, so reporting points in *that*
frame would silently flip x and y under every waypoint in the repo.

Accuracy, measured against ground-truth poses over 24 detections
(`test_agent` runs this every time):

| | lateral error |
| --- | --- |
| box centre with median depth (front surface) | 15.8 mm mean |
| point cloud centroid | 9.4 mm mean |
| **centroid + half the depth extent** | **5.6 mm mean, 6.9 mm max** |

The grasp tolerance is about 28 mm, so the final method leaves plenty of margin — but the
first method does not, which is why the correction is there. The measured object size comes
out at 3.4–3.7 cm against a true 3.6 cm block, a free consistency check on the unprojection.

The landing pad is not detected: it is a task-frame site whose pose is given, and the
observation says so. IK is unchanged from the rest of the repo — damped least-squares on
the pinch site's position and z-axis, 220 iterations, joint limits clamped each step.

## What Claude is and is not told

The observation is the detections and nothing else — pixel box, unprojected centre,
apparent size, camera distance — plus the pad pose and its own pinch point from
proprioception. Never an outcome, never a label, never simulator state. The system prompt deliberately does **not** tell
Claude the gripper's lateral tolerance: judging whether a grasp will hold is the
verifier's job, and an early version that leaked the tolerance made Claude abstain
on offset grasps before the world model ever saw them.

On a repair turn Claude gets its own previous plan, the unchanged observation, and
the verifier's imagined outcome — probabilities, ensemble uncertainty, the imagined
final block position, the likely failure and a suggestion — and is told to change
exactly one thing or abstain.

## Free-form motion is executed but marked unverified

`run_trace` will execute any legal waypoint program, so "move the gripper over the
blue block and hover there" works. The dynamics model was trained on pick-and-place
traces only, so anything whose phase sequence is not the canonical eight is run
**without** a verified prediction and labelled as such in the transcript and the log.
The pick-and-place success test is not reported for those runs either, because it
does not mean anything there.

## The one bug worth knowing about

Every training window was a decoded `.mp4`. Handing the frozen backbone raw
renderer output instead is far enough off-distribution to invert verdicts. On three
scenes whose canonical plan really succeeds:

| observation window | p(success) |
| --- | --- |
| raw renderer frames | 0.17 / 0.30 / 0.12 |
| same frames through h264 | 0.98 / 0.86 / 1.00 |

`verifier.through_codec` round-trips a live window through the codec, and
`encode_live` is the entry point the agent uses. A residual gap remains — an 8-frame
clip is not encoded quite like the first 8 frames of a 48-frame clip, and the
re-encoded window reads slightly more confident — but verdicts agree.

## Measured on live scenes

`test_agent --live 30` renders 30 fresh scenes, aims plans with the same
good/bad mix the dataset generator uses, scores each from pixels, then actually
runs it:

| | live 30 scenes | test split, `docs/results.md` |
| --- | --- | --- |
| verdict agrees with physics | **26/30 = 0.867** | 0.852 |
| false accepts | 2 of 10 real failures | 0.176 |
| false rejects | 2 of 20 real successes | 0.090 |

So the live path is calibrated like the offline evaluation. Both false accepts were
grasp misses, which matches the known weak axis: `lifted` accuracy is 0.720 against
`in_target`'s 0.851. **The loop is much better at catching a bad place target than a
bad grasp.** A deliberately offset grasp is often waved through — the honest summary
is that the world model currently earns its keep on where the block ends up, not on
whether the fingers close on it.

## What the loop actually does, from the logs

Four runs in `results/agent/`, all with Opus 5 (`evidence_reel.mp4` shows the first
three):

| command | loop | outcome |
| --- | --- | --- |
| "pick up the red block and put it on the green pad" | approved at p=0.855 first try | landed 0.022 m from the pad centre, success |
| "balance the red block right on the -x rim of the green pad, overhanging the edge a little" | proposed x=0.390, **rejected at p=0.343** ("block ends outside the landing zone"), repaired to x=0.430, approved at p=0.988 | success |
| "move the gripper over the blue block and hover just above it, do not grab anything" | not verifiable, executed and said so | pinch ends at (0.499, −0.157, 0.241); the blue block was detected at (0.495, −0.178) |
| "put the red block on the green pad, but let go early — release it around x=0.33, y=0.30" | rejected at p=0.000, and Claude declined both repairs — "changed nothing: the verifier's only failure is exactly what the operator asked for" | **nothing executed** |

The repair branch fires less often than the design implies, because Opus 5's first
plan for a plain pick-and-place is usually already right. Where it fires, it fires on
placement, and the second plan is better. The last row is the more interesting
behaviour: given a rejection it disagreed with, Claude refused to silently override the
operator's explicit coordinate, and the loop stopped without executing rather than
pretending to resolve the conflict.

## Limits

- One skill, one block colour that the outcome test cares about, one camera.
- Grasp-failure detection is the weak axis (see above); do not read an approval as a
  guarantee that the fingers will hold.
- Off-distribution place targets pull the imagined final position back toward the
  training distribution, so a plan that drops the block well past the pad can still be
  approved. The verifier is trustworthy near the plans it was trained on.
- The planner is one stateless turn per call; there is no conversation memory between
  commands beyond what the loop passes explicitly.
