"""Closed loop: Claude proposes a skill trace, the world model imagines it, Claude repairs it.

    uv run python -m waddle_wm.agent --instruction "pick up the red block and put it on the green pad"

One command runs perceive -> propose -> verify -> repair/approve -> execute and writes a
replayable JSON log next to the rendered episode. Object positions come from the camera
(`waddle_wm.perception`), the planner is Claude Opus 5, and the verifier is the local
action-conditioned latent world model, which never sees a label and never runs physics.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from waddle_wm.perception import QUERIES, SceneCamera, landing_pad
from waddle_wm.planner import ClaudePlanner, MODEL, Plan, PlanError, SkillStep, describe_observation
from waddle_wm.sim import relling_scene as scene
from waddle_wm.sim.env import FRAMES_TOTAL, PRELUDE_FRAMES, WINDOW_FRAMES, TabletopEnv
from waddle_wm.verifier import Verifier, VerificationResult

DEFAULT_CHECKPOINT = Path("models/multiblock_world_model.pt")
DEFAULT_LOGS = Path("results/agent")
UNVERIFIABLE = ("the world model was trained on pick-and-place traces only, so this free-form "
                "motion is off-distribution and was executed without a verified prediction")


def verifier_skip_reason(step: SkillStep, verifier=None) -> str | None:
    if not step.pick_place_shaped:
        return UNVERIFIABLE
    if verifier is not None and verifier.manifest.get("schema_version", 3) >= 4:
        return None
    if step.object != "red block" or step.destination != "green pad":
        return ("the current world-model checkpoint was trained only on red block → green pad; "
                f"{step.object} → {step.destination} was executed and measured, but not model-verified")
    return None


def verdict_for_claude(result: VerificationResult) -> dict:
    """The verifier's imagined outcome, in the shape the repair prompt shows Claude."""
    return {"imagined_success_probability": round(result.success_probability, 3),
            "ensemble_uncertainty": round(result.uncertainty, 3),
            "p_block_lifted": round(result.lifted_probability, 3),
            "p_block_in_landing_zone": round(result.in_target_probability, 3),
            "imagined_final_block_xy": [round(v, 4) for v in result.predicted_block_xy],
            "likely_failure": result.likely_failure,
            "verifier_suggestion": result.suggestion,
            "approved": result.approve}


@dataclass
class Round:
    index: int
    kind: str                      # "propose" or "repair"
    plan: dict
    verified: bool
    verdict: dict | None = None
    skipped_reason: str | None = None


@dataclass
class AgentRun:
    instruction: str
    seed: int
    model: str
    verifier: str
    observation: str
    scene: dict
    rounds: list[Round] = field(default_factory=list)
    decision: str = "pending"       # executed | abstained | rejected | error
    reason: str = ""
    execution: dict | None = None
    video: str | None = None
    planner_calls: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    frames: object = field(default=None, repr=False)   # rendered episode, kept out of the log
    final_plan: object = field(default=None, repr=False)   # the Plan the loop ended on, for a re-run

    def as_json(self) -> dict:
        carried = self.frames, self.final_plan
        self.frames = self.final_plan = None
        payload = asdict(self)
        self.frames, self.final_plan = carried
        payload.pop("frames")
        payload.pop("final_plan")
        payload["cost_usd"] = round(sum(call.get("cost_usd") or 0.0 for call in self.planner_calls), 4)
        return payload


class SkillAgent:
    """Holds the simulator and the verifier so a UI can drive many instructions in one process."""

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT, seed: int = 0, model: str = MODEL,
                 repairs: int = 2, threshold: float | None = None, verify: bool = True,
                 verifier_mode: str | None = None):
        self.verifier_mode = verifier_mode or ("world-model" if verify else "none")
        if self.verifier_mode not in ("none", "rules", "world-model"):
            raise ValueError(f"unknown verifier mode {self.verifier_mode!r}")
        self.verifier = Verifier(checkpoint, threshold=threshold) if self.verifier_mode == "world-model" else None
        manifest = self.verifier.manifest if self.verifier else {}
        spawn = {key: tuple(manifest[key]) for key in ("block_spawn_low", "block_spawn_high") if key in manifest}
        self.env = TabletopEnv(seed=seed, **spawn)   # sample scenes the world model was trained on
        self.camera = SceneCamera(self.env.model, self.env.data)
        self.planner = ClaudePlanner(model=model)
        self.repairs, self.model = repairs, model

    def observe(self, block_xy=None, target_xy=None):
        """Reset the scene, render the pre-execution window, and encode it once."""
        self.env.reset(block_xy if block_xy is not None else self.env.sample_scene(), target_xy)
        frames = self.env.observation_frames(WINDOW_FRAMES)
        latent = self.verifier.encode_live(frames) if self.verifier else None
        return frames, latent

    def perceive(self):
        """`bounding_box` -> `detect_in_base` for every object the agent looks for."""
        return self.camera.detect_all(QUERIES)

    def ground_step(self, step: SkillStep) -> SkillStep:
        """Refresh moving object coordinates before each step of a compound plan."""
        points = {detection.label: detection.point_base for detection in self.perceive()}
        source = points[step.object]
        destination = points.get(step.destination)
        references = getattr(self, "_plan_points", points)

        def rebase(xy, name):
            planned = np.asarray(xy)
            current = np.asarray(points[name][:2])
            original = np.asarray(references.get(name, points[name])[:2])
            anchor = current if np.linalg.norm(planned - current) <= np.linalg.norm(planned - original) else original
            return planned + current - anchor

        trace = []
        for entry in step.trace:
            entry = dict(entry)
            if entry["phase"] in ("approach", "descend", "lift"):
                entry["target"] = [*rebase(entry["target"][:2], step.object).tolist(), entry["target"][2]]
            elif destination is not None and entry["phase"] in ("move", "place", "retreat"):
                z = float(destination[2] + 2 * scene.BLOCK_HALF) if entry["phase"] == "place" else entry["target"][2]
                entry["target"] = [*rebase(entry["target"][:2], step.destination).tolist(), z]
            trace.append(entry)
        return SkillStep(step.object, step.destination, trace)

    def observe_current(self):
        """Observe the result of one step before verifying the next one."""
        self.env.clear_recording()
        frames = self.env.observation_frames(WINDOW_FRAMES)
        return frames, self.verifier.encode_live(frames) if self.verifier else None

    def rule_verdict(self, step: SkillStep) -> VerificationResult:
        """Deterministic geometry/reachability checks; no learned outcome prediction."""
        points = {d.label: np.asarray(d.point_base) for d in self.perceive()}
        grasp = np.asarray(next(e["target"] for e in step.trace if e["phase"] == "descend"))
        place = np.asarray(next(e["target"] for e in step.trace if e["phase"] == "place"))
        grasp_error = float(np.linalg.norm(grasp[:2] - points[step.object][:2]))
        destination = points.get(step.destination)
        pad, radius = landing_pad(self.env.model)
        target_error = float(np.linalg.norm(place[:2] - (destination[:2] if destination is not None else pad)))
        target_limit = scene.BLOCK_HALF * 1.5 if destination is not None else radius
        height_ok = destination is None or place[2] >= destination[2] + scene.BLOCK_HALF * 1.5
        reachable = True
        try:
            q = np.array([self.env.data.joint(j).qpos[0] for j in scene.ARM_JOINTS])
            for entry in step.trace:
                if "target" in entry:
                    q = self.env._ik(entry["target"], q)
        except RuntimeError:
            reachable = False
        failure = ("grasp misses the block" if grasp_error > 0.028 else
                   "block misses the destination" if target_error > target_limit or not height_ok else
                   "waypoint is unreachable" if not reachable else None)
        suggestion = {"grasp misses the block": "re-aim the grasp at the observed block centre",
                      "block misses the destination": "move the place target onto the destination",
                      "waypoint is unreachable": "move the waypoint inside the arm's reachable workspace"}.get(failure)
        score = 0.0 if failure else 1.0
        return VerificationResult(not failure, score, 0.0, float(grasp_error <= 0.028),
                                  float(target_error <= target_limit and height_ok), place[:2].tolist(),
                                  failure, suggestion)

    def verify_step(self, step: SkillStep, latent):
        if self.verifier_mode == "none":
            return None, "verifier disabled"
        if self.verifier_mode == "rules":
            return self.rule_verdict(step), None
        skip = verifier_skip_reason(step, self.verifier)
        if skip:
            return None, skip
        positions = {d.label.replace(" ", "_"): d.point_base for d in self.perceive()}
        return self.verifier.verify(latent, step.trace, step.object.replace(" ", "_"),
                                    step.destination.replace(" ", "_"), positions), None

    def run(self, instruction: str, seed: int = 0, block_xy=None, target_xy=None, on_event=None) -> AgentRun:
        started = time.time()
        emit = on_event or (lambda *_: None)
        _, latent = self.observe(block_xy, target_xy)
        detections = self.perceive()
        self._plan_points = {detection.label: detection.point_base for detection in detections}
        pad = landing_pad(self.env.model)
        observation = describe_observation(detections, pad, self.env.state()["gripper_pos"])
        run = AgentRun(instruction, seed, self.model, self.verifier_mode, observation,
                       {"detections": [detection.summary() for detection in detections],
                        "landing_pad": {"centre": [round(v, 4) for v in pad[0]], "radius": round(pad[1], 4)},
                        "truth": {name: [round(v, 4) for v in position]
                                  for name, position in self.env.block_positions().items()}})
        emit("observed", {"observation": observation,
                          "detections": [detection.summary() for detection in detections]})

        try:
            plan = self.planner.propose(instruction, observation)
        except (PlanError, RuntimeError) as error:
            run.decision, run.reason = "error", str(error)
            run.planner_calls, run.seconds = self.planner.calls[-4:], time.time() - started
            emit("error", {"reason": run.reason})
            return run

        if len(plan.steps) > 1:
            return self.run_sequence(run, plan, latent, emit, started)

        step = self.ground_step(plan.steps[0]) if plan.steps else None

        for index in range(self.repairs + 1):
            kind = "propose" if index == 0 else "repair"
            emit("plan", {"round": index, "kind": kind, **plan.summary()})
            if not plan.executable:
                run.rounds.append(Round(index, kind, plan.summary(), verified=False,
                                        skipped_reason="planner abstained"))
                run.decision, run.reason = "abstained", plan.note
                emit("abstained", {"reason": plan.note})
                break
            result, skip = self.verify_step(step, latent)
            if skip:
                reason = skip
                run.rounds.append(Round(index, kind, plan.summary(), verified=False, skipped_reason=reason))
                emit("unverified", {"round": index, "reason": reason})
                run.decision, run.reason = "executed", reason
                break
            verdict = verdict_for_claude(result)
            run.rounds.append(Round(index, kind, plan.summary(), verified=True, verdict=verdict))
            emit("verdict", {"round": index, "verifier": self.verifier_mode, **verdict})
            if result.approve:
                run.decision, run.reason = "executed", f"verifier approved at p={result.success_probability:.3f}"
                break
            if index == self.repairs:
                run.decision = "rejected"
                run.reason = (f"still rejected after {self.repairs} repairs "
                              f"(p={result.success_probability:.3f}, {result.likely_failure})")
                emit("rejected", {"reason": run.reason})
                break
            try:
                plan = self.planner.repair(instruction, observation, plan, verdict)
                step = self.ground_step(plan.steps[0]) if plan.steps else None
            except (PlanError, RuntimeError) as error:
                run.decision, run.reason = "error", str(error)
                emit("error", {"reason": run.reason})
                break

        if run.decision == "executed":
            emit("executing", {"object": step.object, "destination": step.destination, "trace": step.summary()["trace"]})
            run.execution, frames = self.execute(step)
            run.execution["frames"] = len(frames)
            emit("executed", run.execution)
            run.frames = frames
        if run.final_plan is None:
            run.final_plan = plan
        run.planner_calls = list(self.planner.calls)
        run.seconds = round(time.time() - started, 2)
        return run

    def run_sequence(self, run: AgentRun, plan: Plan, latent, emit, started) -> AgentRun:
        """Execute a compound instruction one observed, scored, and logged step at a time."""
        executions, clips = [], []
        for index, proposed in enumerate(plan.steps):
            for attempt in range(self.repairs + 1):
                step = self.ground_step(proposed)
                emit("step", {"index": index + 1, "total": len(plan.steps), "repair": attempt,
                              **step.summary()})
                result, skip = self.verify_step(step, latent)
                round_index = index * (self.repairs + 1) + attempt
                if skip is not None:
                    run.rounds.append(Round(round_index, "step", step.summary(), verified=False,
                                            skipped_reason=skip))
                    emit("unverified", {"round": round_index, "reason": skip})
                    break
                verdict = verdict_for_claude(result)
                run.rounds.append(Round(round_index, "step", step.summary(), verified=True, verdict=verdict))
                emit("verdict", {"round": round_index, "verifier": self.verifier_mode, **verdict})
                if result.approve:
                    break
                if attempt == self.repairs:
                    run.decision = "rejected"
                    run.reason = (f"step {index + 1} still rejected after {self.repairs} repairs "
                                  f"at p={result.success_probability:.3f}: {result.likely_failure}")
                    run.final_plan = Plan(plan.intent, "execute", [step], plan.note)
                    emit("rejected", {"reason": run.reason})
                    break
                detections = self.perceive()
                observation = describe_observation(detections, landing_pad(self.env.model),
                                                   self.env.state()["gripper_pos"])
                try:
                    repaired = self.planner.repair(
                        f"Step {index + 1} of the compound task: move the {step.object} onto the {step.destination}",
                        observation, Plan(plan.intent, "execute", [step], plan.note), verdict)
                    if len(repaired.steps) != 1 or repaired.steps[0].object != step.object or repaired.steps[0].destination != step.destination:
                        raise PlanError("a compound-step repair must keep the same object and destination")
                    proposed = repaired.steps[0]
                except (PlanError, RuntimeError) as error:
                    run.decision, run.reason = "error", str(error)
                    emit("error", {"reason": run.reason})
                    break

            if run.decision in ("rejected", "error"):
                break

            emit("executing", {"step": index + 1, "object": step.object,
                                "destination": step.destination, "trace": step.summary()["trace"]})
            execution, frames = self.execute(step)
            executions.append(execution); clips.append(frames)
            emit("executed", execution)
            if execution.get("success") is False:
                run.decision, run.reason = "rejected", f"step {index + 1} failed in MuJoCo: {execution['failure_mode']}"
                break
            if index + 1 < len(plan.steps):
                _, latent = self.observe_current()
        else:
            run.decision, run.reason = "executed", f"completed {len(plan.steps)} sequential steps"

        run.execution = {"steps": executions, "success": run.decision == "executed"}
        run.frames = np.concatenate(clips) if clips else None
        if run.final_plan is None:
            run.final_plan = plan
        run.planner_calls = list(self.planner.calls)
        run.seconds = round(time.time() - started, 2)
        return run

    def replay(self, plan: Plan, block_xy=None):
        """Run a plan on the same scene with no verifier veto — how a rejection really ends."""
        self.env.reset(block_xy if block_xy is not None else self.env.sample_scene())
        self.env.observation_frames(WINDOW_FRAMES)
        return self.execute(self.ground_step(plan.steps[0]))

    def execute(self, step: SkillStep):
        """Run the accepted trace on the real simulator, continuing from the observation window."""
        budget = FRAMES_TOTAL if step.pick_place_shaped else None
        block = step.object.replace(" ", "_")
        destination = step.destination.replace(" ", "_")
        episode = self.env.run_trace(step.trace, frames_total=budget, prelude_frames=PRELUDE_FRAMES,
                                     block=block, destination=destination)
        state = episode.state_after
        result = {"object": step.object, "destination": step.destination,
                  "max_block_z": round(float(state["max_block_z"]), 4),
                  "final_block_xy": [round(v, 4) for v in state["block_pos"][:2]],
                  "final_gripper_xyz": [round(v, 4) for v in state["gripper_pos"]],
                  "target_distance": round(float(state["target_distance"]), 4)}
        if step.pick_place_shaped:
            result = {"success": episode.success, "failure_mode": episode.failure_mode, **result}
        else:
            result["outcome"] = "free-form motion, so the pick-and-place success test does not apply"
        return result, episode.frames


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repairs", type=int, default=2)
    ap.add_argument("--threshold", type=float, help="override the checkpoint's calibrated threshold")
    ap.add_argument("--verifier", choices=("none", "rules", "world-model"), default="world-model")
    ap.add_argument("--no-verify", action="store_true", help="deprecated alias for --verifier none")
    ap.add_argument("--out", type=Path, default=DEFAULT_LOGS)
    ap.add_argument("--block-x", type=float); ap.add_argument("--block-y", type=float)
    args = ap.parse_args()

    mode = "none" if args.no_verify else args.verifier
    agent = SkillAgent(args.checkpoint, args.seed, args.model, args.repairs, args.threshold,
                       verifier_mode=mode)
    block_xy = [args.block_x, args.block_y] if args.block_x is not None else None
    run = agent.run(args.instruction, args.seed, block_xy,
                    on_event=lambda name, payload: print(f"[{name}] {json.dumps(payload, default=float)}", flush=True))

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if run.frames is not None:
        video = args.out / f"{stamp}.mp4"
        iio.imwrite(video, np.asarray(run.frames), fps=10, codec="libx264")
        run.video = str(video)
    log = args.out / f"{stamp}.json"
    log.write_text(json.dumps(run.as_json(), indent=2, default=float))
    print(f"\ndecision={run.decision} ({run.reason})")
    print(f"log {log}" + (f"\nvideo {run.video}" if run.video else ""))


if __name__ == "__main__":
    main()
