# What the verifier is for: a measured answer

The project asked whether an LLM can use a world model to catch bad robot plans before they run.
Until now the answer was a catch rate — how often the verifier rejected a plan it was meant to
reject — and every flaw it was ever shown was a 6 cm aiming error. That number was always going to
look good, and it could not distinguish two very different things: a system that spots broken
plans, and a system that predicts what a plan will do.

This page separates them and reports both. The short version:

> **The verifier is a reliable defect gate and a poor outcome predictor.** It separates broken
> plans from competent ones perfectly — 63 plans, zero overlap, AUC 1.000 — and it does not rank
> outcomes among plans that all look reasonable. That is a narrower claim than "visual world
> model", and unlike the broad claim it is measured, bounded, and deployable today.

Everything below is reproducible from runs committed to this repo. No number here needs a GPU, a
Claude call, or a re-run.

```bash
uv run python -m waddle_wm.analyse_separation
```

---

## 1. The headline: a gate that never misfired

Two populations of opening plans, scored by the world-model verifier before anything moved:

| population | n | p range | mean p |
| --- | --- | --- | --- |
| scripted-flawed (6 cm grasp / 22 cm place offset) | 16 | 0.013 – 0.207 | **0.086** |
| model-authored, unprompted | 47 | 0.441 – 0.981 | **0.849** |

**Separation AUC 1.000.** The highest-scoring broken plan is 0.207. The lowest-scoring competent
plan is 0.441. The distributions do not touch, so **any threshold in [0.21, 0.44] classifies all
63 plans correctly.**

This is the usable result. A gate at `p < 0.21` is a real safety net against the failure mode that
actually bites in practice — a planner emitting a waypoint in the wrong frame, a transposed
coordinate, a unit error — and on this evidence it does not fire spuriously.

## 2. The honest limit: it does not rank outcomes

Restrict to the competent population only — plans a planner produced unprompted, all of which look
plausible — and ask whether a higher probability means a likelier success:

**Ranking AUC 0.070** (43 successes, 4 failures). The four plans that failed scored
**0.944, 0.944, 0.971, 0.971** — at the top of the range.

Read this as *no evidence of ranking ability*, not as a reliable point estimate: four failures is
far too few to measure an AUC. The defensible statement is that nothing here shows `p` predicting
outcomes among plausible plans, and something would have to show it before `p` is treated as a
confidence score.

**Why the distinction matters.** Detecting a defect is not modelling a world. A schema check
detects malformed JSON without modelling JSON. What is demonstrated is detection; prediction is
not, and "world model" is a claim about prediction.

## 3. The number not to quote

Pooled over both populations the AUC is **0.814** — far better than §2 and much more flattering.
It is an artefact.

The scripted sweeps contribute almost only failures at low `p`; the live-plan sweeps almost only
successes at high `p`. A pooled AUC across them mostly measures *which sweep a plan came from*,
which is trivial. This is Simpson's paradox, and it is an easy trap: a single AUC reported over
mixed-difficulty populations will always flatter the model.

It is printed by the analysis script, labelled as not citable, precisely so it does not get
quoted later by someone reading only the summary.

## 4. Planner capability is not the bottleneck

`demo.py --live-plan` makes the planner author its own opening plan instead of being handed a
scripted flaw. The scripted sweep only ever tested whether a model can apply a fix the verifier's
rejection reason already implies — closer to copying a number than to planning. This tests whether
the planner errs unprompted.

Over 24 scenes, unverified, both models:

| planner | working plans | failed on |
| --- | --- | --- |
| Claude Opus 5 | 22/24 | scenes 1, 4, 23 |
| Claude Haiku 4.5 | 22/24 | scenes 1, 4, 23 |

Same rate, and **the identical three scenes**, with identical failure modes. When two models of
very different capability fail on exactly the same inputs, the inputs are what is being measured:
these failures are perception and scene geometry, not planning.

The practical read: **for this class of task a small model is sufficient** — planner capability is
not the limiting factor. The read to avoid: that Haiku equals Opus in general. Both are at ceiling
here, so the experiment has no power to separate them.

*Caveat on cost.* No price comparison should be drawn from these runs. `propose_opening` uses a
throwaway agent whose planner calls are not recorded, so the logged `cost_usd` covers repair calls
only. The instrumentation needs fixing before any cost claim is made.

## 5. Vision still buys nothing measurable

The project's central open question is whether the *visual* verifier beats a coordinate-only
geometry rule. The fairest test yet built (`waddle_wm/demo_ambiguous.py`):

Hold the plan **byte-identical** — grasp aimed 2.0 cm from the red block's detected centre, inside
the rules verifier's tolerance in every condition — and move only the **neighbouring block**. The
plan's numbers never change. Only the image changes. Physics swings from never-works to
always-works.

| neighbour at | red reached pad | rules: p | world-model: caught |
| --- | --- | --- | --- |
| 6.0 cm | 0/10 | 1.000 | 1/10 |
| 7.5 cm | 0/10 | 1.000 | 0/10 |
| 8.5 cm | 5/10 | 1.000 | 0/10 |
| 22.0 cm | 10/10 | 1.000 | 2/10 |

The rules verifier is blind here **by construction** — grasp-to-target distance is 0.020 m in every
row, so its verdict is pinned at 1.000 and it cannot express the difference. The information exists
only in the image.

The world model's probability stayed flat (0.75 – 0.84) across the whole gradient, ranked outcomes
at chance (**AUC 0.472**, n=40), and placed its lowest mean confidence on the condition that always
succeeds. This is consistent with [`results.md`](results.md) §3, where the world model ties a
plan-only control that never sees an image.

Two findings fell out of building it, both worth keeping:

- **The gripper needs ≈7.5 cm of block separation to close at all.** Below that, even a perfect
  grasp fails 12/12. So "the grasp is nearer a neighbour than the target" and "a correct grasp
  would have worked" are disjoint regimes in this environment — a real constraint on what
  wrong-block experiments can test here.
- On a geometrically perfect but physically impossible plan (blocks 4.2 cm apart), the world model
  returned mean **p = 0.960** and approved 12/12. Off-distribution inputs did not raise its
  uncertainty — 0.134 on failures against 0.144 on successes. A verifier that does not know what it
  does not know is the more serious problem, and it is measured here rather than asserted.

## 6. Where the verifier costs more than it saves

A 3-scene smoke test on a two-step instruction (*stack blue on red, then yellow on the pad*):

| arm | Opus | Haiku |
| --- | --- | --- |
| unverified | 3/3 | 3/3 |
| world-model | **2/3** | 3/3 |

In the failing scene the verifier approved step 1 (p=0.891), then rejected step 2 three times
(p=0.340, 0.309, 0.340) and halted with nothing executed — while the unverified arm ran that same
step 2 successfully. A false rejection, consistent with the documented held-out false-reject rate
of 0.679.

N=3 and a single event, so this is a signal to investigate, not a result. It does say the gate in
§1 should be applied at the threshold that was actually measured, rather than trusting the
verifier's judgement near the middle of its range.

## 7. Methods worth reusing

Two pieces of experimental design here transfer beyond this project.

**Counterfactual scoring of a verifier.** A verifier's rejections are normally unscorable: a
rejected plan never runs, so there is no outcome to compare against, and the evaluation set is
silently restricted to plans the verifier already liked. The sweep design removes that — every arm
receives the *identical* opening plan, and the `none` arm executes it regardless of any verdict. So
each scene yields `(probability, what that exact plan then did)` with no selection bias. This was
already being produced by the existing sweeps and had never been used.

**Separating detection from prediction.** Report a verifier's discriminative power on a homogeneous
population, not a pooled one, and report the composition of every population scored. §3 is what
happens when this is skipped.

## 8. What this changes

Before: an unfalsifiable claim about a visual world model, supported by a catch rate on flaws
chosen to be caught.

After, all measured and bounded:

1. A **deployable gate** with a threshold derived from data, perfect on 63 plans.
2. A **stated limit** — the probability is not a confidence score, and the evidence needed to
   promote it to one is named (a task that fails often enough to give ~30 failures).
3. A **negative result on vision**, from a test where the coordinate-only baseline was structurally
   incapable of winning on merit — and it still did not lose.
4. A **cost finding**: this task does not need a frontier planner.
5. Two **reusable evaluation methods**, and one statistical trap documented before it made it into
   a paper.

A well-characterised narrow capability is more useful than a broad claim that does not survive
scrutiny. The narrow one is what ships.

## Limits

Single seed throughout. One task family (tabletop pick-and-place, one block to a pad, plus a
3-scene stacking probe). Two flaw types, both inserted rather than encountered. The ranking figure
in §2 rests on 4 failures. The ambiguous-neighbour conditions in §5 are off-distribution for
`models/multiblock_world_model.pt`, which was trained with blocks ≥7.5 cm apart — an explanation for
that null, though not a defence of the flat uncertainty.

## Reproduce

```bash
# the analysis in §1-§3, from committed runs; no Claude calls, no GPU
uv run python -m waddle_wm.analyse_separation --out docs/verifier_separation.md

# §4: the planner authors its own plan
uv run python -m waddle_wm.demo --sweep 24 --live-plan --arm none --arm world-model \
    --model claude-haiku-4-5-20251001 --out results/demo-live-haiku24 --no-video

# §6: a harder, multi-step instruction through the same machinery
uv run python -m waddle_wm.demo --sweep 3 --live-plan --scenario grasp_miss \
    --instruction "stack the blue block on top of the red block, then put the yellow block on the green pad" \
    --arm none --arm world-model --out results/smoke-stack-haiku --no-video
```

Data: [`results/demo`](../results/demo), [`results/demo-live-haiku24`](../results/demo-live-haiku24),
[`results/demo-live-opus24`](../results/demo-live-opus24),
[`results/smoke-stack-haiku`](../results/smoke-stack-haiku),
[`results/smoke-stack-opus`](../results/smoke-stack-opus).
[`verifier_separation.md`](verifier_separation.md) is the analysis script's generated output.
