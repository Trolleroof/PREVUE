# UR5e Latent-Dynamics Data Schema

The training example is a **latent transition**, not a labelled clip:

```text
observation window (frames)  +  action chunk (skill trace slice)
        -> future observation window latent  (+ grounded state at that latent)
```

Everything below is fixed so that the same record shape serves training, rollout,
and pre-execution verification. Schema version **3** is the red-only corpus;
version **4** extends the same transition contract to all blocks and stacking.

## Schema 4 multi-block extension

Schema 4 keeps the 48-frame grid and 13-d action chunks unchanged. Each record adds
`skill.params.object` and `skill.params.destination`, where the source is red, blue,
or yellow and the destination is the green pad or either other block. The per-frame
`all_block_pos` track has shape `(48, 3, 3)` in fixed red/blue/yellow order.

The grounded readout predicts all block XYZ positions plus the gripper XYZ. Success is
then computed from the selected future block and destination geometry. No color-specific
success head is used. Long instructions are verified one atomic transition at a time,
with a new observation window after every executed step.

## 1. Fixed time grid

Every episode is rendered to exactly `N = 48` frames at `fps = 10` (4.8 s):

| segment | frames | contents |
| --- | --- | --- |
| prelude | `0 .. 7` | arm at home, nothing moving — the *pre-execution observation* |
| execution | `8 ..` | the `pick_place` skill trace |
| tail padding | `.. 47` | idle frames after `retreat`, so `N` is exact |

The grid is chopped into `N / W = 6` non-overlapping **windows** of `W = 8` frames
(0.8 s). Window `k` spans frames `[8k, 8k+7]`; its **anchor** is its last frame
`a_k = 8k + 7`, i.e. `7, 15, 23, 31, 39, 47`.

- **Context length** `C = 8` = one window. A latent is always the embedding of a
  whole window, so context/horizon/stride are the same unit and rollout is just
  window chaining.
- **Horizon** `H = 8` frames = 0.8 s per predictor step.
- **Rollout** from window 0 (the static prelude) to window 5 is 5 chained steps,
  4.0 s of imagined future — the entire skill.

Window 0 is the only window that contains no motion, which is what makes
pre-execution verification well posed: the verifier sees the scene, not the act.

## 2. Episode record — `records.jsonl`

One line per physical execution. `frames_path` is the full 48-frame clip; windows
are slices of it, never separate files.

```json
{
  "schema_version": 3,
  "episode_id": "ur5e_0042",
  "seed": 0,
  "split": "train",
  "observation": {
    "camera": "demo", "frames_path": "clips/ur5e_0042.mp4",
    "width": 256, "height": 256, "fps": 10,
    "frames_total": 48, "prelude_frames": 8, "window_frames": 8,
    "frame_times_s": [0.0, 0.1, "..."]
  },
  "skill": {
    "name": "pick_place",
    "params": {"target_xy": [0.36, 0.24], "grasp_offset_xy": [0.021, -0.004]},
    "trace": [
      {"phase": "approach", "target": [0.39, -0.19, 0.24], "frames": [8, 19]},
      {"phase": "close", "value": 255.0, "frames": [20, 25]}
    ]
  },
  "tracks": {
    "phase": [0, 0, "...", 8],
    "waypoint": [[0.30, -0.10, 0.35], "..."],
    "gripper": [0.0, "..."],
    "pinch_pos": [[0.30, -0.10, 0.35], "..."],
    "block_pos": [[0.39, -0.19, 0.018], "..."],
    "max_block_z": [0.018, "..."],
    "target_distance": [0.52, "..."]
  },
  "state_before": {"block_pos": ["..."], "...": "..."},
  "state_after": {"block_pos": ["..."], "...": "..."},
  "outcome": {"success": false, "failure_mode": "target_miss"}
}
```

`tracks.*` are per-frame arrays of length `frames_total`, aligned index-for-index
with the rendered clip. They are produced by the same MuJoCo stepping loop that
renders the frames, so alignment is exact rather than interpolated.

Each `trace` entry now carries `frames: [first, last]`, the inclusive frame span
during which that phase was commanded. This is what makes a skill trace
compilable into per-frame actions (§4).

## 3. Action encoding

The action at frame `t` is a 13-d vector — what was *commanded*, never what was
observed:

```text
a_t = [ phase one-hot (9) | commanded pinch waypoint xyz (3) | gripper cmd (1) ]
```

Phases are ordered `idle, approach, descend, close, lift, move, place, open,
retreat`; `idle` covers the prelude and the tail padding. `close`/`open` hold the
previous waypoint. `gripper` is `0.0` open / `1.0` closed. Waypoints are absolute
MuJoCo world metres and are normalised with dataset statistics before use.

An **action chunk** for a transition is the `H = 8` per-frame actions executed
*during the target window*:

```text
chunk_k = [a_t for t in window k+1] -> shape (8, 13)
```

No motor torques and no joint angles appear anywhere: every element of `a_t` is
recoverable from a skill trace alone, which is the requirement for using it at
plan time.

## 4. Transition record — the actual training example

Built by `waddle_wm.windows`, not stored as a separate corpus; it is an index
over `records.jsonl` plus the embedding cache.

```json
{
  "episode_id": "ur5e_0042", "split": "train", "step": 2,
  "context": {"window": 2, "frames": [16, 23], "anchor": 23},
  "action_chunk": {"frames": [24, 31], "length": 8},
  "target":  {"window": 3, "frames": [24, 31], "anchor": 31},
  "horizon_frames": 8, "horizon_s": 0.8
}
```

Model contract:

```text
z_k      = Enc(frames[8k .. 8k+7])            frozen, cached  (1024-d)
z_hat    = z_k + f(z_k, chunk_k)              trainable predictor
target   = z_{k+1}                            frozen, cached
```

`Enc` is the frozen V-JEPA 2 ViT-L trunk, mean-pooled over tokens. `f` is the only
thing trained on latents. Rollout is `z_hat` fed back into `f`.

### Targets

| target | source | used for |
| --- | --- | --- |
| `z_{k+1}` | embedding cache | latent regression — the primary loss |
| `block_pos[3]` at `a_{k+1}` | `tracks.block_pos` | grounded readout |
| `pinch_pos[3]` at `a_{k+1}` | `tracks.pinch_pos` | grounded readout |
| `lifted` at `a_{k+1}` | `tracks.max_block_z > 0.09` | grounded readout |
| `in_target` at `a_{k+1}` | `tracks.target_distance <= 0.105` | grounded readout |
| `outcome.success` | episode | terminal readout only (window 5) |

Grounded readouts are decoded **from a latent** by a separate head `g(z)`, trained
on real latents only. Verification applies `g` to an *imagined* latent, so the
readout never sees a predicted latent during training and the reported
verification numbers are honest.

`success` on the current corpus is exactly `lifted AND in_target` at the terminal
anchor, so it is not given its own head; it is read off the terminal readout.

## 5. Plan-time compilation

At verification time there is no execution, so there are no `tracks`. A skill
trace is compiled into action chunks with per-phase frame durations taken from the
training split (`manifest.json -> phase_frames`, the per-phase median):

```text
compile_plan(trace, phase_frames) -> (48, 13) action array -> 5 chunks of (8, 13)
```

Training uses the *executed* per-frame actions; verification uses *compiled* ones.
The two differ whenever a phase runs long or short, and that gap is real model
error, so verifier metrics are always reported with compiled chunks.

## 6. Scene and outcome variation

`generate_dataset` perturbs two independent things so that neither the plan nor
the scene alone determines the outcome:

- `target_xy` — the place target; displaced off the landing zone on ~45% of
  episodes, producing `target_miss`.
- `grasp_offset_xy` — a lateral offset applied to the approach/descend/lift
  waypoints, producing a genuine `missed` grasp when it exceeds the finger span.

Both are recorded in `skill.params` and are visible in the trace waypoints, so a
verifier that only reads the plan can still separate `target_miss`, but it cannot
separate `missed` without looking at where the block actually is. Positions are
metres in MuJoCo world coordinates; quaternions are `[w, x, y, z]`.

## 7. On-disk layout

```text
data/ur5e_wm/
  records.jsonl            episode records (schema 3, with tracks)
  clips/ur5e_XXXX.mp4      48 frames, 256x256, 10 fps
  manifest.json            grid constants, home_waypoint, phase_frames
  window_embeddings.pt     {episode_id: (6, 1024) float32} frozen V-JEPA latents
```

Normalisation statistics are *not* in the manifest — they are fitted on the train
split and saved inside the model checkpoint, so a checkpoint carries everything
needed to score a plan.

`data/ur5e_tabletop/` is the schema-2 corpus kept for reference; it has no
prelude, no tracks, and no grasp failures, and cannot express a transition record.
