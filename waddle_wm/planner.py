"""Claude Opus 5 planner: natural-language command -> structured skill trace.

The planner is the only non-deterministic part of the agent. It never touches
MuJoCo and never sees an outcome label: it is given the task instruction, a
summary of what the camera observes, and — on a repair turn — the verifier's
imagined outcome, and it answers with a waypoint program in the same format
`waddle_wm.actions.compile_plan` and `TabletopEnv.run_trace` already consume.

Claude is reached through the `claude` CLI in headless mode, so this uses the
Claude Code login already on the machine rather than a separate API key.

    uv run python -m waddle_wm.planner --instruction "put the red block on the green pad"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

from waddle_wm.actions import GRIPPER_VALUES, PHASES
from waddle_wm.sim.env import GRASP_Z, GRIPPER_CLOSED, GRIPPER_OPEN, HOVER_Z, TRANSIT_Z

MODEL = "claude-opus-5"
MOVE_PHASES = tuple(phase for phase in PHASES if phase not in ("idle", *GRIPPER_VALUES))
PICK_PLACE_SHAPE = ("approach", "descend", "close", "lift", "move", "place", "open", "retreat")
WORKSPACE = {"x": (0.20, 0.72), "y": (-0.50, 0.50), "z": (0.012, 0.45)}
MAX_PHASES = 8

SYSTEM_PROMPT = f"""You are the high-level planner for a UR5e arm with a Robotiq 2F-85 gripper on a tabletop.
You turn one natural-language command into a waypoint program. You do not run physics and you do not
judge success — a separate deterministic world-model verifier does that and may send the plan back to you.

Reply with ONE JSON object and nothing else. No prose, no markdown fence.

{{"intent": "<one line restating the command in your own words>",
  "action": "execute" | "abstain",
  "trace": [{{"phase": "<phase>", "target": [x, y, z]}}, ...],
  "note": "<one or two sentences: what you aimed at, or what you changed and why>"}}

Rules:
- Phases, in this order where used: {", ".join(MOVE_PHASES)} for motion, plus "close"/"open" for the gripper.
- A motion phase carries "target": [x, y, z] in metres, world frame. "close"/"open" carry no target.
- At most {MAX_PHASES} phases. Every phase must move the arm somewhere sensible; do not pad.
- Workspace limits: x {WORKSPACE['x']}, y {WORKSPACE['y']}, z {WORKSPACE['z']}. Targets outside are rejected.
- To pick an object: approach above it, descend to z={GRASP_Z}, close, lift, move, place, open, retreat.
  Aim approach/descend/lift at the object's observed x,y unless the operator asks for a different grasp
  point. You do not know the gripper's tolerances — propose the plan and let the verifier judge it.
- Hover height is {HOVER_Z}, transit height is {TRANSIT_Z}. Only descend to {GRASP_Z} when grasping or placing.
- "abstain" with an empty trace if the command is unsafe, out of the workspace, or refers to something
  that is not in the observation. Say why in "note".
- On a repair turn you are given the verifier's imagined outcome for your own plan. Change exactly one
  thing — the grasp waypoint, the place target, or abstain — and say which in "note"."""

GRIPPER_LIMITS = {"open": GRIPPER_OPEN, "close": GRIPPER_CLOSED}


class PlanError(ValueError):
    """A malformed or out-of-bounds plan. The message is fed back to Claude verbatim."""


@dataclass
class Plan:
    intent: str
    action: str
    trace: list[dict]
    note: str
    raw: str = ""

    @property
    def executable(self) -> bool:
        return self.action == "execute" and bool(self.trace)

    @property
    def pick_place_shaped(self) -> bool:
        """True when the trace matches the shape the verifier's dynamics were trained on."""
        return tuple(entry["phase"] for entry in self.trace) == PICK_PLACE_SHAPE

    def summary(self) -> dict:
        return {"intent": self.intent, "action": self.action, "note": self.note,
                "trace": [{"phase": e["phase"], **({"target": [round(v, 4) for v in e["target"]]} if "target" in e else {})}
                          for e in self.trace]}


def validate(payload: dict) -> Plan:
    """Claude's JSON -> a Plan, or `PlanError` with a message Claude can act on."""
    for key in ("intent", "action", "trace", "note"):
        if key not in payload:
            raise PlanError(f"missing key {key!r}; the reply must have intent, action, trace, note")
    if payload["action"] not in ("execute", "abstain"):
        raise PlanError(f"action must be 'execute' or 'abstain', got {payload['action']!r}")
    trace, cleaned = payload["trace"], []
    if not isinstance(trace, list):
        raise PlanError("trace must be a list of phase objects")
    if payload["action"] == "abstain":
        trace = []
    if not trace and payload["action"] == "execute":
        raise PlanError("action 'execute' needs a non-empty trace")
    if len(trace) > MAX_PHASES:
        raise PlanError(f"trace has {len(trace)} phases, the arm accepts at most {MAX_PHASES}")
    for i, entry in enumerate(trace):
        phase = entry.get("phase") if isinstance(entry, dict) else None
        if phase not in PHASES or phase == "idle":
            raise PlanError(f"trace[{i}]: phase must be one of {MOVE_PHASES + tuple(GRIPPER_LIMITS)}, got {phase!r}")
        if phase in GRIPPER_LIMITS:
            cleaned.append({"phase": phase, "value": GRIPPER_LIMITS[phase]})
            continue
        target = entry.get("target")
        if not (isinstance(target, (list, tuple)) and len(target) == 3):
            raise PlanError(f"trace[{i}] ({phase}): needs \"target\": [x, y, z]")
        try:
            point = [float(v) for v in target]
        except (TypeError, ValueError):
            raise PlanError(f"trace[{i}] ({phase}): target must be three numbers, got {target!r}")
        for value, axis in zip(point, "xyz"):
            low, high = WORKSPACE[axis]
            if not low <= value <= high:
                raise PlanError(f"trace[{i}] ({phase}): {axis}={value:.3f} is outside the workspace {low}..{high}")
        cleaned.append({"phase": phase, "target": point})
    return Plan(str(payload["intent"]), payload["action"], cleaned, str(payload["note"]))


def parse(text: str) -> Plan:
    """Claude's raw stdout -> a validated Plan."""
    stripped = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise PlanError(f"reply was not JSON: {text[:200]!r}")
    try:
        payload = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError as error:
        raise PlanError(f"reply was not valid JSON ({error}); send one JSON object and nothing else")
    return validate(payload)


def describe_observation(detections, pad, gripper_xyz) -> str:
    """The scene as the planner sees it: detections from the camera, never simulator state.

    Each line is one `bounding_box` -> `detect_in_base` result, so what Claude reads is what
    the perception primitives measured — pixel box, unprojected centre, apparent size.
    """
    lines = []
    for detection in detections:
        point = detection.point_base
        lines.append(f"- {detection.label}: centre x={point[0]:.3f}, y={point[1]:.3f}, z={point[2]:.3f}"
                     f" (box {detection.box} px, {detection.size_m * 100:.1f} cm across, {detection.depth_m:.2f} m from the camera)")
    pad_xy, pad_radius = pad
    lines.append(f"- green landing pad (known task frame, not detected) centred at "
                 f"x={pad_xy[0]:.3f}, y={pad_xy[1]:.3f}, radius {pad_radius:.3f}")
    lines.append(f"- gripper pinch point (proprioception) at x={gripper_xyz[0]:.3f}, "
                 f"y={gripper_xyz[1]:.3f}, z={gripper_xyz[2]:.3f}")
    return "\n".join(lines)


def propose_prompt(instruction: str, observation: str) -> str:
    return (f"Task instruction (from the operator):\n{instruction}\n\n"
            f"Camera observation of the scene right now:\n{observation}\n\n"
            "Propose the waypoint program. Reply with the JSON object only.")


def repair_prompt(instruction: str, observation: str, plan: Plan, verdict: dict) -> str:
    lines = [f"Your previous plan for: {instruction}", json.dumps(plan.summary(), indent=2), "",
             "Scene observation (unchanged):", observation, "",
             "The world-model verifier imagined this plan being executed and reported:"]
    lines.append(json.dumps(verdict, indent=2))
    lines.append("")
    lines.append("Change exactly one thing to raise the imagined success probability, or abstain. "
                 "Reply with the JSON object only.")
    return "\n".join(lines)


def error_prompt(error: str) -> str:
    return f"Your last reply was rejected: {error}\nSend a corrected JSON object and nothing else."


@dataclass
class ClaudePlanner:
    """`claude -p` in headless mode, one stateless turn per call."""

    model: str = MODEL
    timeout: float = 180.0
    retries: int = 2
    binary: str = field(default_factory=lambda: shutil.which("claude") or "claude")
    calls: list[dict] = field(default_factory=list)

    def complete(self, prompt: str) -> str:
        command = [self.binary, "-p", prompt, "--output-format", "json", "--model", self.model,
                   "--system-prompt", SYSTEM_PROMPT, "--allowed-tools", "", "--strict-mcp-config",
                   "--disable-slash-commands", "--max-turns", "1"]
        environment = {**os.environ, "CLAUDE_CODE_ENTRYPOINT": "waddle-planner"}
        finished = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout,
                                  env=environment, cwd=os.getcwd())
        if finished.returncode != 0:
            raise RuntimeError(f"claude exited {finished.returncode}: {finished.stderr.strip()[:400]}")
        envelope = json.loads(finished.stdout)
        if envelope.get("is_error"):
            raise RuntimeError(f"claude reported an error: {str(envelope.get('result'))[:400]}")
        self.calls.append({"model": self.model, "cost_usd": envelope.get("total_cost_usd"),
                           "duration_ms": envelope.get("duration_ms"), "session_id": envelope.get("session_id")})
        return envelope["result"]

    def plan(self, prompt: str) -> Plan:
        """Ask once, and re-ask with the rejection reason if the reply does not validate."""
        attempt, message = prompt, ""
        for _ in range(self.retries + 1):
            raw = self.complete(attempt)
            try:
                plan = parse(raw)
                plan.raw = raw
                return plan
            except PlanError as error:
                message = str(error)
                attempt = f"{prompt}\n\n{error_prompt(message)}"
        raise PlanError(f"Claude did not return a usable plan after {self.retries + 1} attempts: {message}")

    def propose(self, instruction: str, observation: str) -> Plan:
        return self.plan(propose_prompt(instruction, observation))

    def repair(self, instruction: str, observation: str, plan: Plan, verdict: dict) -> Plan:
        return self.plan(repair_prompt(instruction, observation, plan, verdict))


def main():
    ap = argparse.ArgumentParser(description="One Claude proposal for the current simulator scene")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from waddle_wm.perception import QUERIES, SceneCamera, landing_pad
    from waddle_wm.sim.env import TabletopEnv

    env = TabletopEnv(seed=args.seed)
    env.reset(env.sample_scene())
    camera = SceneCamera(env.model, env.data)
    observation = describe_observation(camera.detect_all(QUERIES), landing_pad(env.model), env.home_waypoint())
    plan = ClaudePlanner(model=args.model).propose(args.instruction, observation)
    print(json.dumps({"instruction": args.instruction, **plan.summary(),
                      "pick_place_shaped": plan.pick_place_shaped}, indent=2, default=float))


if __name__ == "__main__":
    main()
