# skill-level-world-model

A UR5e arm does tabletop pick-and-place in MuJoCo. Claude writes the plan as a structured
waypoint program. Before anything moves, a learned **verifier** encodes the scene with a frozen
V-JEPA 2 trunk, rolls the proposed plan forward in latent space, and predicts what it will do —
so a plan that would fail can be rejected and handed back to Claude for one targeted repair.

The research question is narrower than "build a world model":

> Does *imagining the plan visually* catch failures that a coordinate-only geometry rule misses?

Everything here is measured against that question, including the parts where the answer is no.
The docs state what is not established as plainly as what is; this page does the same.

---

## The result so far

**Verification is worth having.** On identical scenes handed an identical, deliberately flawed
opening plan: 0/8 succeed unverified, 6/8 with the learned verifier, 7/8 with the geometry rule
([`docs/demo.md`](docs/demo.md) §2). The loop — camera → Claude → imagined outcome → repair →
approval → physics — is real end to end, with no privileged simulator state in the decision path.

**The learned verifier is a reliable defect gate.** Over 63 opening plans, scripted-flawed plans
score p ∈ [0.013, 0.207] and model-authored plans p ∈ [0.441, 0.981] — **separation AUC 1.000, no
overlap**, so any threshold in [0.21, 0.44] classifies all 63 correctly
([`docs/verifier_characterisation.md`](docs/verifier_characterisation.md) §1).

**It is not a calibrated outcome predictor.** Restricted to plans that all look plausible, ranking
AUC is 0.070 on 43 successes and 4 failures — far too few failures to estimate, so read it as *no
evidence of ranking ability*, not as a measured negative. The probability is not a confidence
score. A pooled AUC of 0.814 is also printed, and is an artefact of mixing populations; do not
quote it.

**Vision has not beaten coordinates on the main line of work.** The deterministic `rules`
verifier, which never sees an image, beats the learned model on the demo sweep (0.88 vs 0.75),
ties a plan-only control that sees no image offline ([`docs/results.md`](docs/results.md) §3), and
was not beaten on the selector benchmark's `visible_omitted_by_coordinates` slice
([`docs/results.md`](docs/results.md) §8). The one demo scene where pixels *should* have won — a
19.4 mm perception error — is a scene the visual verifier also got wrong.

**One place vision does win, on a corpus built so that it must.** A yaw-aware verifier on a task
suite whose blocks are longer than the gripper's jaw stroke beats a same-architecture no-vision
control by **+0.217 AUC** on the coordinate-matched orientation slice, and beats the geometry rule
by 8.4 points of accuracy overall ([`docs/task_suite_world_model.md`](docs/task_suite_world_model.md)
§6). Two caveats travel with it and are stated there: the corpus was constructed so heading is
decisive, and the heading pathway is a fitted probe rather than something the model discovered on
its own. It shows a visual verifier *can* read what coordinates omit; it does not show that
orientation decides real picks that often.

Known weak axes, in the docs' own words: grasp-failure detection (`lifted` accuracy 0.720 on the
main checkpoint), a false-reject rate of 0.679 on held-out multi-block data, an imagined block
position ~0.2 m out (read the probability, never the coordinate), and a single seed throughout.

---

## Quickstart

Python ≥3.11 and [uv](https://docs.astral.sh/uv/). `uv run` resolves the environment on first use.

```bash
git clone https://github.com/Trolleroof/skill-level-world-model
cd skill-level-world-model
```

**Start here — no checkpoint, no GPU, no Claude call, no cost.** This is the headline measurement,
recomputed from sweeps committed to the repo:

```bash
uv run python -m waddle_wm.analyse_separation
```

It prints the separation AUC, the ranking AUC, and the pooled figure it explicitly tells you not
to cite. `--out docs/verifier_separation.md` writes the same report to a file.

Also free — re-verify and re-execute the recorded demo traces without calling Claude, and see how
far the replayed verdicts drift from the recorded ones (0.0000 on the traces in this repo):

```bash
uv run python -m waddle_wm.demo --replay results/demo --arm none --arm rules
```

### What needs what

`models/` is **gitignored**. Nothing in it ships with the repo, so any path marked *checkpoint*
below needs one you train or supply, plus the frozen encoder at
`models/vjepa2-vitl-fpc64-256/` (V-JEPA 2 ViT-L, ~1.3 GB — see
[`docs/backbone_decision.md`](docs/backbone_decision.md)).

| command | Claude calls | checkpoint |
| --- | --- | --- |
| `analyse_separation` | no | no |
| `demo --replay … --arm none --arm rules` | no | no |
| `demo --arm none --arm rules` | yes | no |
| `demo` (all three arms) | yes | yes |
| `server` / `agent` with `--verifier none` or `rules` | yes | no |
| `server` / `agent` with `--verifier world-model` (default) | yes | yes |

Claude is invoked through the `claude` CLI already logged in on the machine, not an API key
([`docs/llm_agent.md`](docs/llm_agent.md)). `--model claude-haiku-4-5-20251001` selects a smaller
planner, which on this task family produced working plans at the same rate as Opus 5 and failed on
the identical scenes ([`docs/verifier_characterisation.md`](docs/verifier_characterisation.md) §4).
No cost comparison is quoted here: the sweeps' cost accounting omits the propose call entirely, so
the repo cannot currently measure what a run costs (§4).

### The full demo

The end-to-end run: one flawed plan, three verifiers (`none`, `rules`, `world-model`), the same
scene, real physics. About 40 s and $0.19 on Opus 5. Traces, videos, and a generated report land
in `results/demo/`.

```bash
uv run python -m waddle_wm.demo                       # all three arms; needs a checkpoint
uv run python -m waddle_wm.demo --arm none --arm rules  # no checkpoint needed
uv run python -m waddle_wm.demo --sweep 8             # the rate table, ~6 min and ~$1.67
```

The `none` arm executes the flawed plan unverified, which is what makes this a claim rather than
an anecdote: the outcome the other arms avoided is measured, not asserted.

### The browser demo

```bash
uv run python -m waddle_wm.server                     # defaults to --verifier world-model
uv run python -m waddle_wm.server --verifier rules    # no checkpoint needed
```

The episode plays on the left, a chat bar on the right, and each step of the loop streams into the
transcript as it happens. Headless equivalent:

```bash
uv run python -m waddle_wm.agent --instruction "put the red block on the green pad"
```

### Training a checkpoint

Commands and their measured outcomes are in [`docs/llm_agent.md`](docs/llm_agent.md#multi-block-verifier-status),
[`docs/results.md`](docs/results.md) and [`docs/task_suite_world_model.md`](docs/task_suite_world_model.md) §4.
The multi-block checkpoint the demo defaults to:

```bash
uv run python -m waddle_wm.train_multiblock_world_model \
  --data data/ur5e_wm_multiblock --out models/multiblock_world_model.pt
```

---

## Repo map

| module | what it does |
| --- | --- |
| `waddle_wm/sim/` | MuJoCo UR5e + Robotiq tabletop environment, dataset generators, and their validators |
| `waddle_wm/perception.py` | `bounding_box`, `detect_in_base`, `approach_until` — the planner reads the camera, never simulator state |
| `waddle_wm/planner.py` | Claude proposes and repairs a structured waypoint program, via the `claude` CLI |
| `waddle_wm/actions.py` | The one action encoding shared by simulator, trainer, and verifier |
| `waddle_wm/verifier.py` | Encode the scene, roll the plan forward, decode an imagined outcome + uncertainty |
| `waddle_wm/suite_verifier.py` | The yaw-aware, multi-subtask serving path; says *which* step it expects to break |
| `waddle_wm/agent.py` | The closed loop: perceive → propose → verify → repair → execute, plus the CLI |
| `waddle_wm/server.py` | The browser demo — same loop, streamed into a page |
| `waddle_wm/demo.py` | Three verifier arms, one identical flawed plan, one command |
| `waddle_wm/demo_ambiguous.py` | The fairest vision-vs-coordinates test built: hold the plan byte-identical, move only the neighbour |
| `waddle_wm/analyse_separation.py` | Detection vs prediction, from committed sweeps; no GPU, no Claude |
| `waddle_wm/program.py`, `pools.py`, `counterfactual.py`, `selectors.py`, `benchmark_selectors.py` | The offline program-ranking benchmark: bounded candidates, frozen pools, every candidate executed from one identical state, three selectors compared |
| `waddle_wm/train_*.py`, `embed_windows.py`, `report_*.py` | Embedding, training, and metric export |
| `waddle_wm/ui/`, `build_experiment_page.py`, `render_demo_clips.py` | The standalone scrubbable results page and its MuJoCo clips |

---

## Where to read more

| doc | the question it answers |
| --- | --- |
| [`docs/project.md`](docs/project.md) | What is being claimed, what exists, and what the architecture is |
| [`docs/verifier_characterisation.md`](docs/verifier_characterisation.md) | What is the verifier's probability actually good for? (gate: yes; confidence score: no) |
| [`docs/demo.md`](docs/demo.md) | Does the closed loop work end to end — and §4, what does it *not* prove? |
| [`docs/results.md`](docs/results.md) | The offline benchmark: dynamics, verifier accuracy, where the information is lost, and the selector comparison |
| [`docs/task_suite_world_model.md`](docs/task_suite_world_model.md) | The one setting where the visual verifier beats every no-vision control, and why that corpus was built that way |
| [`docs/llm_agent.md`](docs/llm_agent.md) | How Claude is prompted, where 3D positions come from, and the plan format |
| [`docs/transition_schema.md`](docs/transition_schema.md) | The training data contract |
| [`docs/backbone_decision.md`](docs/backbone_decision.md) | Why a frozen V-JEPA 2 ViT-L trunk and not V-JEPA 2-AC |
| [`docs/program_schema.md`](docs/program_schema.md) | What a candidate program is, and how pools are frozen |
| [`docs/counterfactual_execution.md`](docs/counterfactual_execution.md) | How every candidate is executed from one identical simulator state |
| [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) | What a locked run records, how the oracle orders candidates, what the validator refuses |
| [`docs/selector_benchmark.md`](docs/selector_benchmark.md) | The design of Claude self-rank vs coordinates vs vision |
| [`docs/feedback_benchmark.md`](docs/feedback_benchmark.md) | The paired repair benchmark and its two budget views |
| [`docs/verifier_separation.md`](docs/verifier_separation.md) | Generated output of `analyse_separation` |
