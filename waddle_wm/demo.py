"""The end-to-end demo: one flawed plan, three verifiers, the same scene, real physics.

    uv run python -m waddle_wm.demo

Each scenario hands **the same** deliberately flawed opening plan to three arms that differ only
in what verifies it — nothing, the deterministic rules, or the learned visual world model — and
then runs whatever each arm decided to run in MuJoCo. The `none` arm is what makes the demo a
claim rather than an anecdote: it executes the flawed plan unverified, so the outcome the other
arms avoided is measured rather than asserted.

The loop per arm is the one in [`agent.py`](agent.py):

    given plan -> verifier imagines it -> Claude repairs one thing -> verifier approves -> MuJoCo

Claude is the only non-deterministic part, so every plan it returns is written into the trace and

    uv run python -m waddle_wm.demo --replay results/demo

re-verifies and re-executes those exact plans with no Claude calls, and reports how far the
replayed verdicts and outcomes drift from the recorded ones. That is the reproduction path for
someone who does not want to spend Opus tokens to see the result.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from waddle_wm.agent import DEFAULT_CHECKPOINT, SkillAgent
from waddle_wm.perception import landing_pad
from waddle_wm.planner import MODEL, Plan, validate
from waddle_wm.sim.env import GRASP_Z, HOVER_Z, TRANSIT_Z

DEFAULT_OUT = Path("results/demo")
ARMS = ("none", "rules", "world-model")


@dataclass(frozen=True)
class Scenario:
    """One scene, one flawed opening plan, and what the flaw is supposed to test."""

    name: str
    instruction: str
    block_xy: tuple[float, float]
    grasp_offset: tuple[float, float]      # metres added to the *observed* block centre
    place_offset: tuple[float, float]      # metres added to the pad centre
    flaw: str
    tests: str

    def opening_plan(self, block_xy, pad_xy, block: str = "red block",
                     grasp_offset=None, place_offset=None, note=None) -> Plan:
        """The flawed plan, aimed off the block or off the pad by the offsets above.

        Built from the camera's estimate of the block, not from simulator state, so the arms are
        given exactly the kind of plan the planner itself emits.
        """
        gx, gy = grasp_offset if grasp_offset is not None else self.grasp_offset
        px, py = place_offset if place_offset is not None else self.place_offset
        grasp = [round(block_xy[0] + gx, 4), round(block_xy[1] + gy, 4)]
        place = [round(pad_xy[0] + px, 4), round(pad_xy[1] + py, 4)]
        trace = [{"phase": "approach", "target": [*grasp, HOVER_Z]},
                 {"phase": "descend", "target": [*grasp, GRASP_Z]},
                 {"phase": "close"},
                 {"phase": "lift", "target": [*grasp, HOVER_Z]},
                 {"phase": "move", "target": [*place, TRANSIT_Z]},
                 {"phase": "place", "target": [*place, GRASP_Z]},
                 {"phase": "open"},
                 {"phase": "retreat", "target": [*place, HOVER_Z]}]
        return validate({"intent": self.instruction, "action": "execute",
                         "note": note if note is not None else self.flaw,
                         "steps": [{"object": block, "destination": "green pad", "trace": trace}]})


# Frozen before the run. Both offsets are large enough to fail in MuJoCo and small enough that
# nothing but a look at the scene distinguishes them from a good plan on paper.
SCENARIOS = (
    Scenario("grasp_miss", "pick up the red block and put it on the green pad",
             block_xy=(0.42, -0.20), grasp_offset=(0.06, 0.0), place_offset=(0.0, 0.0),
             flaw="grasp aimed 6 cm past the block in +x",
             tests="the weak axis: does the imagined future notice the fingers will close on nothing?"),
    Scenario("place_miss", "pick up the red block and put it on the green pad",
             block_xy=(0.42, -0.20), grasp_offset=(0.0, 0.0), place_offset=(0.0, -0.22),
             flaw="release point 22 cm short of the pad in -y",
             tests="the strong axis: the failure is computable from the plan alone."),
)


def demo_scenes(count: int, spread: float = 0.075) -> list[tuple[float, float]]:
    """`count` reproducible red-block positions, clear of the blue and yellow blocks.

    One scene is an anecdote. The sweep runs the same two flaws over several scenes so the results
    page can quote a rate rather than a story. The other two blocks do not move between scenes, so
    a sampled position that lands on one of them would change the task rather than the scene.
    """
    from waddle_wm.sim import relling_scene as scene

    rng = np.random.default_rng(0)
    fixed = [xy for name, xy in zip(scene.BLOCK_NAMES, scene.BLOCK_SPAWN) if name != "red_block"]
    scenes = []
    while len(scenes) < count:
        xy = rng.uniform((0.32, -0.26), (0.66, -0.10))
        if all(np.linalg.norm(xy - np.asarray(other)) >= spread for other in fixed):
            scenes.append((round(float(xy[0]), 4), round(float(xy[1]), 4)))
    return scenes


@dataclass
class RecordedPlanner:
    """Replays the plans Claude returned during the recorded run, in order, without calling it."""

    repairs: list[Plan]
    model: str = MODEL
    calls: list[dict] = field(default_factory=list)

    def propose(self, instruction, observation):
        raise RuntimeError("replay always starts from the recorded opening plan")

    def repair(self, instruction, observation, plan, verdict) -> Plan:
        if not self.repairs:
            raise RuntimeError("the recorded run has no further repair to replay")
        return self.repairs.pop(0)


def recorded_plans(trace: dict) -> tuple[Plan, list[Plan]]:
    """A saved trace -> the opening plan and the repairs Claude produced, as `Plan` objects."""
    plans = [validate(round_["plan"]) for round_ in trace["rounds"]]
    return plans[0], plans[1:]


def verdict_rows(run) -> list[dict]:
    """Every round of the loop, flattened for the report: what was imagined, and about what."""
    rows = []
    for round_ in run.rounds:
        step = round_.plan["steps"][0] if round_.plan.get("steps") else None
        grasp = next((e["target"] for e in step["trace"] if e["phase"] == "descend"), None) if step else None
        place = next((e["target"] for e in step["trace"] if e["phase"] == "place"), None) if step else None
        rows.append({"round": round_.index, "kind": round_.kind, "grasp_xy": grasp[:2] if grasp else None,
                     "place_xy": place[:2] if place else None, "note": round_.plan.get("note", ""),
                     "verdict": round_.verdict, "skipped_reason": round_.skipped_reason,
                     "plan": round_.plan})   # the whole plan, so `--replay` can re-run this round
    return rows


def run_arm(scenario: Scenario, arm: str, checkpoint: Path, seed: int, model: str, repairs: int,
            planner=None, quiet: bool = False, block_xy=None):
    """One scenario against one verifier arm, from the shared opening plan through MuJoCo."""
    block_xy = list(block_xy or scenario.block_xy)
    agent = SkillAgent(checkpoint, seed=seed, model=model, repairs=repairs, verifier_mode=arm)
    agent.env.reset(block_xy)                         # cheap pre-pass: perceive without encoding,
    detections = {d.label: d.point_base for d in agent.perceive()}   # so the plan is aimed by camera
    pad_xy, _ = landing_pad(agent.env.model)
    opening = scenario.opening_plan(detections["red block"], pad_xy)
    if planner is not None:
        agent.planner = planner
    emit = (lambda *_: None) if quiet else (
        lambda name, payload: print(f"  [{arm}/{name}] {json.dumps(payload, default=float)[:220]}", flush=True))
    run = agent.run(scenario.instruction, seed, block_xy, on_event=emit, opening_plan=opening)
    return run, opening


def arm_record(scenario: Scenario, arm: str, run, opening: Plan, block_xy=None) -> dict:
    execution = run.execution or {}
    return {"scenario": scenario.name, "arm": arm, "instruction": scenario.instruction,
            "flaw": scenario.flaw, "tests": scenario.tests,
            "block_xy": list(block_xy or scenario.block_xy), "scene": run.scene,
            "opening_plan": opening.summary(), "rounds": verdict_rows(run),
            "decision": run.decision, "reason": run.reason,
            "executed_success": execution.get("success"), "failure_mode": execution.get("failure_mode"),
            "final_block_xy": execution.get("final_block_xy"),
            "target_distance": execution.get("target_distance"),
            "max_block_z": execution.get("max_block_z"),
            "seconds": run.seconds, "cost_usd": run.as_json()["cost_usd"],
            "planner_calls": len(run.planner_calls)}


def outcome_word(record: dict) -> str:
    """What actually happened to the block, in one word, or why nothing did."""
    if record["decision"] in ("rejected", "abstained", "error"):
        return f"not executed ({record['decision']})"
    if record["executed_success"] is None:
        return "executed, unscored"
    return "success" if record["executed_success"] else f"failure ({record['failure_mode']})"


def report(records: list[dict]) -> str:
    """The one results table: per scenario, what each arm imagined and what physics then did."""
    lines = ["# Demo results", "",
             "Generated by `uv run python -m waddle_wm.demo`. Every row of a scenario starts from the",
             "identical scene and the identical flawed opening plan; the arms differ only in what",
             "verifies it. `p` is the imagined success probability of the plan that was finally run",
             "(the opening plan for `none`, which verifies nothing).", ""]
    for name in dict.fromkeys(record["scenario"] for record in records):
        rows = [record for record in records if record["scenario"] == name]
        scenario = next(s for s in SCENARIOS if s.name == name)
        lines += [f"## {name} — {scenario.flaw}", "", scenario.tests, "",
                  "| verifier | opening verdict | repairs | final verdict | MuJoCo outcome | block ended |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for record in rows:
            opening, final = record["rounds"][0], record["rounds"][-1]

            def cell(round_):
                verdict = round_["verdict"]
                if not verdict:
                    return f"_{round_['skipped_reason'] or 'none'}_"
                return (f"p={verdict['imagined_success_probability']:.3f} "
                        f"u={verdict['ensemble_uncertainty']:.3f} "
                        f"{'approve' if verdict['approved'] else 'reject'}"
                        + (f" — {verdict['likely_failure']}" if verdict["likely_failure"] else ""))

            landed = ("—" if record["final_block_xy"] is None else
                      f"({record['final_block_xy'][0]:.3f}, {record['final_block_xy'][1]:.3f}), "
                      f"{record['target_distance']:.3f} m from the pad")
            lines.append(f"| {record['arm']} | {cell(opening)} | {len(record['rounds']) - 1} | "
                         f"{cell(final)} | {outcome_word(record)} | {landed} |")
        lines.append("")
    return "\n".join(lines)


def sweep_report(records: list[dict], scenes: list) -> str:
    """The rate table: how often each arm ended with the block on the pad, over every scene."""
    lines = ["# Demo sweep", "",
             f"The same two flawed opening plans over {len(scenes)} scenes, all three arms, "
             "identical scenes across arms.",
             "`caught` counts scenes where the verifier rejected the flawed opening plan;",
             "`success` counts scenes where the block finished on the pad in MuJoCo.", "",
             "| scenario | verifier | caught | repaired to success | success rate |",
             "| --- | --- | --- | --- | --- |"]
    for scenario in dict.fromkeys(record["scenario"] for record in records):
        for arm in ARMS:
            rows = [r for r in records if r["scenario"] == scenario and r["arm"] == arm]
            if not rows:
                continue
            caught = sum(bool(r["rounds"][0]["verdict"]) and not r["rounds"][0]["verdict"]["approved"]
                         for r in rows)
            wins = sum(r["executed_success"] is True for r in rows)
            lines.append(f"| {scenario} | {arm} | {caught}/{len(rows)} | {wins}/{len(rows)} | "
                         f"{wins / len(rows):.2f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--replay", type=Path, help="re-run the traces in this directory without calling Claude")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repairs", type=int, default=2)
    ap.add_argument("--scenario", action="append", choices=[s.name for s in SCENARIOS])
    ap.add_argument("--arm", action="append", choices=ARMS)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--sweep", type=int, metavar="SCENES",
                    help="run every scenario and arm over this many scenes and report rates")
    args = ap.parse_args()

    scenarios = [s for s in SCENARIOS if not args.scenario or s.name in args.scenario]
    arms = [arm for arm in ARMS if not args.arm or arm in args.arm]
    started, records, drifts = time.time(), [], []

    if args.sweep:
        return run_sweep(args, scenarios, arms, started)

    for scenario in scenarios:
        for arm in arms:
            print(f"\n=== {scenario.name} / {arm} ===", flush=True)
            planner = None
            if args.replay:
                saved = json.loads((args.replay / f"{scenario.name}.{arm}.json").read_text())
                planner = RecordedPlanner(recorded_plans(saved)[1])
            run, opening = run_arm(scenario, arm, args.checkpoint, args.seed, args.model, args.repairs,
                                   planner=planner)
            record = arm_record(scenario, arm, run, opening)
            records.append(record)
            print(f"  -> {record['decision']}: {outcome_word(record)}", flush=True)

            # A replay re-runs the recorded plans, so its frames *are* the recorded run's frames —
            # worth writing, since the videos are gitignored and this path costs no tokens.
            video_dir = args.replay if args.replay else args.out
            if run.frames is not None and not args.no_video:
                video_dir.mkdir(parents=True, exist_ok=True)
                iio.imwrite(video_dir / f"{scenario.name}.{arm}.mp4", np.asarray(run.frames),
                            fps=10, codec="libx264")
            if args.replay:
                drifts += replay_drift(saved, record)
                continue
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{scenario.name}.{arm}.json").write_text(json.dumps(record, indent=2, default=float))

    text = report(records)
    if args.replay:
        worst = max((d["delta"] for d in drifts), default=0.0)
        mismatched = [d for d in drifts if d["kind"] == "outcome" and d["delta"]]
        print("\n" + text)
        print(f"replayed {len(records)} arms with no Claude calls; largest probability drift "
              f"{worst:.4f}, {len(mismatched)} outcome mismatches")
        (args.replay / "replay_report.md").write_text(text)
        return

    (args.out / "report.md").write_text(text)
    (args.out / "summary.json").write_text(json.dumps(
        {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": args.model, "seed": args.seed,
         "checkpoint": str(args.checkpoint), "repairs": args.repairs,
         "seconds": round(time.time() - started, 1),
         "cost_usd": round(sum(record["cost_usd"] for record in records), 4),
         "arms": records}, indent=2, default=float))
    print("\n" + text)
    print(f"traces {args.out}/  report {args.out / 'report.md'}")


def run_sweep(args, scenarios, arms, started) -> None:
    """The same flaws over many scenes, so the results page can quote a rate instead of a story."""
    scenes, records = demo_scenes(args.sweep), []
    for index, block_xy in enumerate(scenes):
        for scenario in scenarios:
            for arm in arms:
                run, opening = run_arm(scenario, arm, args.checkpoint, args.seed, args.model,
                                       args.repairs, quiet=True, block_xy=block_xy)
                record = arm_record(scenario, arm, run, opening, block_xy=block_xy)
                record["scene_index"] = index
                records.append(record)
                print(f"scene {index} ({block_xy[0]:.3f}, {block_xy[1]:.3f}) {scenario.name}/{arm}: "
                      f"{outcome_word(record)}", flush=True)

    text = sweep_report(records, scenes)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sweep.md").write_text(text)
    (args.out / "sweep.json").write_text(json.dumps(
        {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": args.model, "scenes": scenes,
         "seconds": round(time.time() - started, 1),
         "cost_usd": round(sum(record["cost_usd"] for record in records), 4),
         "arms": records}, indent=2, default=float))
    print("\n" + text)
    print(f"sweep {args.out / 'sweep.md'}")


def replay_drift(saved: dict, fresh: dict) -> list[dict]:
    """How far a replayed arm moved from what was recorded — probabilities and the MuJoCo outcome."""
    drifts = []
    for old, new in zip(saved["rounds"], fresh["rounds"]):
        if old["verdict"] and new["verdict"]:
            delta = abs(old["verdict"]["imagined_success_probability"]
                        - new["verdict"]["imagined_success_probability"])
            drifts.append({"kind": "verdict", "round": old["round"], "delta": delta})
            print(f"  replay round {old['round']}: p {old['verdict']['imagined_success_probability']:.3f} "
                  f"-> {new['verdict']['imagined_success_probability']:.3f}")
    same = saved["executed_success"] == fresh["executed_success"] and saved["decision"] == fresh["decision"]
    drifts.append({"kind": "outcome", "delta": 0.0 if same else 1.0})
    if not same:
        print(f"  replay MISMATCH: recorded {saved['decision']}/{saved['executed_success']}, "
              f"replayed {fresh['decision']}/{fresh['executed_success']}")
    return drifts


if __name__ == "__main__":
    main()
