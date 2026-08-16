# Self-Consistent World-Model Agent for Robot Skill Verification

## One-sentence pitch

> We build an LLM robot agent that uses a self-consistent visual world model to simulate proposed skills, detect likely failures before execution, and revise its plan.

## Core idea

**Skill-level world model: an LLM-driven evaluator for robot plans**

```text
LLM agent proposes a robot skill
        ↓
world model predicts future robot/object states or frames
        ↓
verifier predicts success, failure mode, and uncertainty
        ↓
LLM revises the plan or executes it
```

For the current UR5e project, the concrete version is:

```text
camera frames + task instruction
        → LLM proposes pick/place trace
        → action-conditioned world model rolls out the trace
        → predict:
             - will the object be lifted?
             - will it reach the target?
             - what failure will occur?
        → LLM adjusts grasp or target
```

This is closer to [SC3](https://weichengtseng.github.io/sc3-eval/) than a simple success classifier because the model still evaluates an imagined future conditioned on actions. The classifier can be the first implementation, but the research claim is:

> Can an LLM use an action-conditioned visual world model to identify and repair robot plans before execution?

## Why this framing

The important evaluation is not just “does the classifier predict success?” It is:

- Does the agent avoid failed grasps?
- Does it repair a bad grasp waypoint?
- Does it choose a better plan than the original LLM?
- Does it reduce real/simulated failed executions?
- Does uncertainty cause it to request another view or stop?

Do not start by training a giant video model. The strongest project at this stage is a **small, inspectable world-model agent** that demonstrates closed-loop improvement:

```text
bad plan → imagined failure → LLM repair → improved plan
```

That is a clearer demo and research question than “we trained a world model”: an agent that creates and improves robot behavior from task instructions.

## MVP

- One UR5e pick-and-place task.
- One or two camera views initially.
- LLM outputs a structured skill trace.
- Frozen V-JEPA encodes the initial observation.
- Small action-conditioned latent model predicts future states.
- Outcome head predicts `lifted`, `in_target`, `success`, and failure mode.
- LLM receives the prediction and revises the plan.
- Compare:
  - LLM without world model.
  - LLM plus plan-only success predictor.
  - LLM plus visual world-model verifier.

## What already exists in this repo

| component | status |
| --- | --- |
| MuJoCo UR5e + Robotiq tabletop simulator | done |
| Repeatable `pick_place` dataset generation | done |
| JSONL transition schema with frames, skill params, state, outcome | done |
| Frozen V-JEPA embedding pipeline | done |
| Action-conditioned latent rollout | done |
| Verifier (`lifted`, `in_target`, failure mode, uncertainty) | done |
| Perception primitives (`bounding_box`, `detect_in_base`, `approach_until`) | done |
| Claude Opus 5 planner and repair loop | done |
| Browser demo with a chat bar | done |
| Bounded code-as-policy program schema and cached Claude candidate pools | done |
| Benchmark artifact schema, oracle ordering, and integrity validator | done |
| Counterfactual execution of every candidate from one identical state | done |
| End-to-end demo: one flawed plan, three verifiers, one command | done |

See [`transition_schema.md`](transition_schema.md) for the data contract, [`results.md`](results.md) for measured verifier performance, [`backbone_decision.md`](backbone_decision.md) for the frozen-backbone choice, [`llm_agent.md`](llm_agent.md) for the planner, the perception primitives, and the repair loop, [`program_schema.md`](program_schema.md) for the candidate format the program-ranking benchmark uses, [`benchmark_protocol.md`](benchmark_protocol.md) for what a locked run records, how the oracle orders candidates, and what the validator refuses, [`counterfactual_execution.md`](counterfactual_execution.md) for how the outcome records that answer key is built from are produced — every candidate executed from one identical simulator snapshot, and [`demo.md`](demo.md) for the reproducible end-to-end demo and the honest reading of what it does and does not establish.

## Architecture

```text
camera frames + task instruction
        |
        v
   LLM planner
   outputs structured skill trace
        |
        v
frozen V-JEPA encodes initial observation window
        |
        v
action-conditioned latent dynamics
rolls out compiled trace (5 steps, 4.0 s imagined)
        |
        v
outcome head + uncertainty
(lifted, in_target, success, failure_mode)
        |
        v
   LLM repair loop
   revise grasp / target / waypoint, or execute
```

The world model stays at the **skill-planning layer** — it does not belong in the low-level servo loop.

## Evaluation baselines

| condition | what it tests |
| --- | --- |
| LLM only | raw plan quality without simulation |
| LLM + plan-only predictor | can action params alone predict success? |
| LLM + visual world-model verifier | does imagined future catch failures the plan hides? |

Success is measured by reduced failed executions and improved plans after repair, not classifier accuracy alone.

## SC3 extensions (future)

| SC3 idea | This project |
| --- | --- |
| Forward dynamics | Predict future visual/latent states from the skill trace |
| Inverse dynamics | Check whether imagined frames are consistent with commanded actions |
| Cross-view consistency | Predict the same future event across external and wrist cameras |
| Test-time consistency | Stop or revise when predicted actions and imagined frames disagree |

## Next steps

Done — see [`llm_agent.md`](llm_agent.md):

1. ~~**LLM planner**~~ — Claude Opus 5 emits a structured skill trace from detections + task instruction.
2. ~~**Verifier integration**~~ — every pick-and-place plan is scored before execution.
3. ~~**Repair loop**~~ — imagined failure and uncertainty go back to Claude, one change at a time.

Done — see [`demo.md`](demo.md):

4. ~~**Three-way comparison**~~ — no verifier vs rules vs visual world model, identical scenes and
   identical flawed opening plan, one command. Verification converts 0/8 into 6/8 and 7/8; the
   *visual* half of the claim is still unsupported, because the geometry rule wins.

Still open:

5. **Uncertainty behavior** — request another view or halt when ensemble disagreement is high.
   Partly there: the loop halts after two failed repairs, and [`demo.md`](demo.md) §2 shows what
   that costs — one scene where the learned verifier refuses a plan that would have worked.
6. **Grasp-failure detection** — the weak axis. `lifted` accuracy is 0.720 against `in_target`'s
   0.851, so an offset grasp is often approved. This is what the agent needs next, not more planning.
7. **Skills above primitives** — a middle layer that lets the agent name and reuse parametrised
   skills instead of emitting one flat trace per command. The bounded program schema in
   [`program_schema.md`](program_schema.md) is the first half of this: Claude picks the strategy
   and bounded parameters, the compiler keeps the waypoints.
8. **Program ranking** — with candidate pools cached, execute every candidate from an identical
   MuJoCo snapshot and compare Claude self-rank, the estimated-state heuristic, and the visual
   world model against the oracle.
