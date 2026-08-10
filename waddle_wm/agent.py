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
from waddle_wm.planner import ClaudePlanner, MODEL, Plan, PlanError, describe_observation
from waddle_wm.sim.env import FRAMES_TOTAL, PRELUDE_FRAMES, WINDOW_FRAMES, TabletopEnv
from waddle_wm.verifier import Verifier, VerificationResult

DEFAULT_CHECKPOINT = Path("models/latent_dynamics_wide.pt")
DEFAULT_LOGS = Path("results/agent")
UNVERIFIABLE = ("the world model was trained on pick-and-place traces only, so this free-form "
                "motion is off-distribution and was executed without a verified prediction")


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

    def as_json(self) -> dict:
        frames, self.frames = self.frames, None
        payload = asdict(self)
        self.frames = frames
        payload.pop("frames")
        payload["cost_usd"] = round(sum(call.get("cost_usd") or 0.0 for call in self.planner_calls), 4)
        return payload


class SkillAgent:
    """Holds the simulator and the verifier so a UI can drive many instructions in one process."""

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT, seed: int = 0, model: str = MODEL,
                 repairs: int = 2, threshold: float = 0.5, verify: bool = True):
        self.verifier = Verifier(checkpoint, threshold=threshold) if verify else None
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

    def run(self, instruction: str, seed: int = 0, block_xy=None, target_xy=None, on_event=None) -> AgentRun:
        started = time.time()
        emit = on_event or (lambda *_: None)
        _, latent = self.observe(block_xy, target_xy)
        detections = self.perceive()
        pad = landing_pad(self.env.model)
        observation = describe_observation(detections, pad, self.env.state()["gripper_pos"])
        run = AgentRun(instruction, seed, self.model, observation,
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

        for index in range(self.repairs + 1):
            kind = "propose" if index == 0 else "repair"
            emit("plan", {"round": index, "kind": kind, **plan.summary()})
            if not plan.executable:
                run.rounds.append(Round(index, kind, plan.summary(), verified=False,
                                        skipped_reason="planner abstained"))
                run.decision, run.reason = "abstained", plan.note
                emit("abstained", {"reason": plan.note})
                break
            if self.verifier is None or not plan.pick_place_shaped:
                reason = UNVERIFIABLE if self.verifier else "verifier disabled"
                run.rounds.append(Round(index, kind, plan.summary(), verified=False, skipped_reason=reason))
                emit("unverified", {"round": index, "reason": reason})
                run.decision, run.reason = "executed", reason
                break
            result = self.verifier.verify(latent, plan.trace)
            verdict = verdict_for_claude(result)
            run.rounds.append(Round(index, kind, plan.summary(), verified=True, verdict=verdict))
            emit("verdict", {"round": index, **verdict})
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
            except (PlanError, RuntimeError) as error:
                run.decision, run.reason = "error", str(error)
                emit("error", {"reason": run.reason})
                break

        if run.decision == "executed":
            emit("executing", {"trace": plan.summary()["trace"]})
            run.execution, frames = self.execute(plan)
            run.execution["frames"] = len(frames)
            emit("executed", run.execution)
            run.frames = frames
        run.planner_calls = list(self.planner.calls)
        run.seconds = round(time.time() - started, 2)
        return run

    def execute(self, plan: Plan):
        """Run the accepted trace on the real simulator, continuing from the observation window."""
        budget = FRAMES_TOTAL if plan.pick_place_shaped else None
        episode = self.env.run_trace(plan.trace, frames_total=budget, prelude_frames=PRELUDE_FRAMES)
        state = episode.state_after
        result = {"max_block_z": round(float(state["max_block_z"]), 4),
                  "final_block_xy": [round(v, 4) for v in state["block_pos"][:2]],
                  "final_gripper_xyz": [round(v, 4) for v in state["gripper_pos"]],
                  "target_distance": round(float(state["target_distance"]), 4)}
        if plan.pick_place_shaped:      # the lift/in-zone test only means something for a pick and place
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
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-verify", action="store_true", help="LLM-only baseline: skip the world model")
    ap.add_argument("--out", type=Path, default=DEFAULT_LOGS)
    ap.add_argument("--block-x", type=float); ap.add_argument("--block-y", type=float)
    args = ap.parse_args()

    agent = SkillAgent(args.checkpoint, args.seed, args.model, args.repairs, args.threshold, verify=not args.no_verify)
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
