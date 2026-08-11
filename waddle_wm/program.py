"""Bounded code-as-policy programs: the candidate format every selector ranks.

A *program* is a short symbolic policy — `detect("red block")`, then hover, descend,
grasp, lift, move, place, release — with bounded numeric parameters. It carries no
simulator coordinates: every waypoint is expressed as an offset from something the
camera found, so the same program means different waypoints in different scenes and
Claude cannot smuggle ground truth into a candidate.

    program  --ground(observation)-->  Plan / SkillStep  --compile_plan-->  actions
                                                          --run_trace---->  MuJoCo

Grounding is deterministic and is the only place coordinates appear. The grounded plan
goes through `planner.validate`, so a program can never produce a trace the existing
UR5e path would refuse: same phases, same workspace, same phase budget.

    uv run python -m waddle_wm.program --seed 0        # canonical program, grounded on one scene
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field

from waddle_wm import planner
from waddle_wm.planner import DESTINATIONS, MAX_PHASES, OBJECTS, PlanError, SkillStep

SCHEMA_VERSION = 1

# The whole API a candidate may call. Anything else is rejected before grounding.
OPS = ("detect", "move_above", "descend_to", "grasp", "lift", "release", "retreat", "on_failure", "abort")
QUERIES = (*OBJECTS, "green pad")
DIRECTIONS = {"top": (0.0, 0.0), "+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}
RETRY_POLICIES = ("abort", "redetect_regrasp")

# Bounded parameter ranges, in millimetres. These are the knobs Claude gets to choose;
# a candidate outside them is a static reject, not a plan the verifier has to judge.
RANGES = {
    "offset_mm": (-60.0, 60.0),          # lateral offset from the detected centre, per axis
    "standoff_mm": (0.0, 60.0),          # how far the hover point leads the descent
    "hover_mm": (40.0, 400.0),           # move_above height above the table
    "lift_mm": (40.0, 400.0),            # lift / retreat height above the table
    "grasp_mm": (12.0, 60.0),            # descend_to height before the grasp
    "release_mm": (12.0, 140.0),         # descend_to height after the grasp
    "yaw_deg": (-90.0, 90.0),            # wrist heading; the jaws are symmetric, so +-90 covers it
    "max_attempts": (0, 2),              # bounded retries; unbounded retry is a reject
}
MAX_OPS = 16
MAX_REASON = 200
# The diagnostic suite names both halves, so a failure slice is interpretable: these are the
# strategies a scene might call for, and these are the bugs a verifier ought to catch.
STRATEGIES = ("correct", "redetect_regrasp", "orientation_aware_grasp", "alternate_approach",
              "offset_grasp", "controlled_release", "abort_on_uncertainty")
FAULTS = ("stale_coordinates", "bad_grasp", "missing_lift", "early_release", "high_release", "wrong_target")


class ProgramError(ValueError):
    """A malformed, out-of-range, or ungroundable program. The message goes back to Claude verbatim."""


@dataclass
class SceneObservation:
    """What a program may resolve a symbol against: detections and the task-frame pad, nothing else.

    Built from `waddle_wm.perception` in the pool generator, and from plain numbers in the
    offline checks so the schema can be exercised without MuJoCo.
    """

    points: dict[str, list[float]]        # query -> [x, y, z] metres, command frame
    pad_radius: float = 0.105
    seed: int = 0
    text: str = ""
    detections: list[dict] = field(default_factory=list)

    @property
    def observation_id(self) -> str:
        """Stable id for 'the scene the candidates were generated against'."""
        payload = {"seed": self.seed,
                   "points": {k: [round(v, 5) for v in point] for k, point in sorted(self.points.items())},
                   "pad_radius": round(self.pad_radius, 5)}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def point(self, query: str) -> list[float]:
        if query not in self.points:
            raise ProgramError(f"detect({query!r}) found nothing; the scene contains "
                               f"{sorted(self.points)}")
        return list(self.points[query])

    @classmethod
    def from_perception(cls, detections, pad, gripper_xyz, seed: int, text: str = "") -> "SceneObservation":
        pad_xy, pad_radius = pad
        points = {d.label: list(d.point_base) for d in detections}
        points["green pad"] = [float(pad_xy[0]), float(pad_xy[1]), 0.0]
        points["gripper"] = [float(v) for v in gripper_xyz]
        return cls(points, float(pad_radius), seed, text, [d.summary() for d in detections])


@dataclass
class Program:
    """One candidate policy, symbolic. `raw` keeps whatever Claude actually said."""

    strategy: str
    object: str
    destination: str
    ops: list[dict]
    note: str = ""
    raw: str = ""

    def as_json(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "strategy": self.strategy,
                "task": {"object": self.object, "destination": self.destination},
                "ops": self.ops, "note": self.note}

    @property
    def aborts(self) -> str | None:
        """The reason this candidate declines to act, or None if it acts.

        Declining is a first-class candidate: on a scene where nothing in the pool works,
        the honest answer is to stop, and a selector that never gets to choose it cannot be
        credited for avoiding a crash.
        """
        for op in self.ops:
            if op["op"] == "abort":
                return op["reason"]
        return None

    @property
    def retry(self) -> dict:
        for op in self.ops:
            if op["op"] == "on_failure":
                return {"policy": op["policy"], "max_attempts": op["max_attempts"]}
        return {"policy": "abort", "max_attempts": 0}

    @property
    def redetects(self) -> list[int]:
        """Indices of `detect` ops that rebind a symbol after motion has already started."""
        moved, indices = False, []
        for index, op in enumerate(self.ops):
            if op["op"] in ("move_above", "descend_to", "grasp", "lift", "release", "retreat"):
                moved = True
            elif op["op"] == "detect" and moved:
                indices.append(index)
        return indices


@dataclass
class GroundedProgram:
    """A program plus the exact plan it compiles to on one observation."""

    program: Program
    step: SkillStep | None            # None when the program declines to act
    observation_id: str

    @property
    def trace(self) -> list[dict]:
        return self.step.trace if self.step else []

    @property
    def aborts(self) -> str | None:
        return self.program.aborts

    def as_json(self) -> dict:
        return {"program": self.program.as_json(), "observation_id": self.observation_id,
                "grounded_trace": self.step.summary()["trace"] if self.step else [],
                "pick_place_shaped": bool(self.step and self.step.pick_place_shaped),
                "aborts": self.aborts,
                "retry": self.program.retry, "redetect_ops": self.program.redetects}

    def dedup_key(self) -> str:
        """Two candidates are the same candidate when they would behave the same way.

        That is the grounded waypoints (to a tenth of a millimetre), the retry policy, and
        whether a symbol is rebound mid-program — not the symbolic spelling, which Claude
        varies freely for plans that mean exactly the same thing.
        """
        waypoints = [[entry["phase"], *[round(v, 4) for v in entry.get("target", [])],
                      None if entry.get("yaw") is None else round(entry["yaw"], 4)] for entry in self.trace]
        payload = {"task": [self.program.object, self.program.destination], "waypoints": waypoints,
                   "aborts": bool(self.aborts),
                   "retry": self.program.retry, "redetect": bool(self.program.redetects)}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- static validation


def _number(value, name: str, bounds: tuple[float, float]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgramError(f"{name} must be a number, got {value!r}")
    low, high = bounds
    if not low <= float(value) <= high:
        raise ProgramError(f"{name}={value} is outside the allowed range {low}..{high}")
    return float(value)


def _offset(value, name: str) -> list[float]:
    if value is None:
        return [0.0, 0.0]
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise ProgramError(f"{name} must be [dx, dy] in millimetres, got {value!r}")
    return [_number(v, f"{name}[{i}]", RANGES["offset_mm"]) for i, v in enumerate(value)]


def validate_program(payload: dict) -> Program:
    """Claude's JSON -> a Program, or `ProgramError` with a message Claude can act on.

    Static only: schema, known API, parameter ranges, and structure. Whether the grounded
    waypoints are reachable or sensible is decided later, by grounding and by the verifiers.
    """
    if not isinstance(payload, dict):
        raise ProgramError("a program must be one JSON object")
    version = payload.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ProgramError(f"schema_version must be {SCHEMA_VERSION}, got {version!r}")
    task = payload.get("task")
    if not isinstance(task, dict):
        raise ProgramError('missing "task": {"object": ..., "destination": ...}')
    object_name, destination = task.get("object"), task.get("destination")
    if object_name not in OBJECTS:
        raise ProgramError(f"task.object must be one of {OBJECTS}, got {object_name!r}")
    if destination not in DESTINATIONS or destination == object_name:
        raise ProgramError("task.destination must be the green pad or a different block")
    strategy = payload.get("strategy", "pick_place")
    if not isinstance(strategy, str) or not strategy:
        raise ProgramError("strategy must be a short string naming the approach")

    ops = payload.get("ops")
    if not isinstance(ops, list) or not ops:
        raise ProgramError("ops must be a non-empty list of operations")
    if len(ops) > MAX_OPS:
        raise ProgramError(f"program has {len(ops)} ops, at most {MAX_OPS} are allowed")

    bound: set[str] = set()
    cleaned, held, seen_failure, aborted = [], False, False, False
    for index, raw in enumerate(ops):
        where = f"ops[{index}]"
        if not isinstance(raw, dict):
            raise ProgramError(f"{where} must be an object")
        name = raw.get("op")
        if name not in OPS:
            raise ProgramError(f"{where}: unknown operation {name!r}; the API is {OPS}")
        if seen_failure:
            raise ProgramError("on_failure must be the last operation")
        if aborted:
            raise ProgramError("abort must be the last operation; nothing runs after it")

        if name == "detect":
            query, symbol = raw.get("query"), raw.get("as")
            if query not in QUERIES:
                raise ProgramError(f"{where}: detect() accepts {QUERIES}, got {query!r}")
            if not isinstance(symbol, str) or not symbol.isidentifier():
                raise ProgramError(f'{where}: detect() needs "as": "<name>" to bind the result')
            bound.add(symbol)
            cleaned.append({"op": "detect", "query": query, "as": symbol})

        elif name in ("move_above", "descend_to"):
            symbol = raw.get("ref")
            if symbol not in bound:
                raise ProgramError(f"{where}: ref {symbol!r} was never bound by a detect()")
            offset = _offset(raw.get("offset_mm"), f"{where}.offset_mm")
            yaw = raw.get("yaw_deg")
            if yaw is not None:
                yaw = _number(yaw, f"{where}.yaw_deg", RANGES["yaw_deg"])
            if name == "move_above":
                height = _number(raw.get("height_mm", 240.0), f"{where}.height_mm", RANGES["hover_mm"])
                direction = raw.get("direction", "top")
                if direction not in DIRECTIONS:
                    raise ProgramError(f"{where}: direction must be one of {tuple(DIRECTIONS)}, got {direction!r}")
                standoff = _number(raw.get("standoff_mm", 0.0), f"{where}.standoff_mm", RANGES["standoff_mm"])
                cleaned.append({"op": name, "ref": symbol, "offset_mm": offset, "height_mm": height,
                                "direction": direction, "standoff_mm": standoff, "yaw_deg": yaw})
            else:
                bounds = RANGES["release_mm"] if held else RANGES["grasp_mm"]
                height = _number(raw.get("height_mm", 15.0), f"{where}.height_mm", bounds)
                cleaned.append({"op": name, "ref": symbol, "offset_mm": offset, "height_mm": height,
                                "yaw_deg": yaw})

        elif name in ("lift", "retreat"):
            height = _number(raw.get("height_mm", 240.0), f"{where}.height_mm", RANGES["lift_mm"])
            cleaned.append({"op": name, "height_mm": height})

        elif name == "grasp":
            if held:
                raise ProgramError(f"{where}: the gripper is already closed")
            held = True
            cleaned.append({"op": "grasp"})

        elif name == "release":
            if not held:
                raise ProgramError(f"{where}: release() before grasp()")
            held = False
            cleaned.append({"op": "release"})

        elif name == "abort":
            if cleaned and any(op["op"] != "detect" for op in cleaned):
                raise ProgramError(f"{where}: abort() cannot follow motion; a program either acts "
                                   f"or declines, and only detect() may precede it")
            reason = raw.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ProgramError(f'{where}: abort() needs "reason": "<why you decline>"')
            aborted = True
            cleaned.append({"op": "abort", "reason": reason.strip()[:MAX_REASON]})

        else:  # on_failure
            policy = raw.get("policy", "abort")
            if policy not in RETRY_POLICIES:
                raise ProgramError(f"{where}: policy must be one of {RETRY_POLICIES}, got {policy!r}")
            attempts = raw.get("max_attempts", 0)
            if not isinstance(attempts, int) or isinstance(attempts, bool):
                raise ProgramError(f"{where}.max_attempts must be a whole number; retries are bounded")
            attempts = int(_number(attempts, f"{where}.max_attempts", RANGES["max_attempts"]))
            seen_failure = True
            cleaned.append({"op": "on_failure", "policy": policy, "max_attempts": attempts})

    if not aborted and not any(op["op"] == "grasp" for op in cleaned):
        raise ProgramError("a pick-and-place program must grasp() the object, or abort() with a reason")
    if held:
        raise ProgramError("the program ends still holding the object; add release()")
    return Program(strategy, object_name, destination, cleaned, str(payload.get("note", "")))


def parse(text: str) -> Program:
    """Claude's raw stdout -> a validated Program."""
    stripped = text.strip()
    for fence in ("```json", "```"):
        if stripped.startswith(fence):
            stripped = stripped[len(fence):].removesuffix("```").strip()
            break
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise ProgramError(f"reply was not JSON: {text[:200]!r}")
    try:
        payload = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError as error:
        raise ProgramError(f"reply was not valid JSON ({error}); send one JSON object and nothing else")
    program = validate_program(payload)
    program.raw = text
    return program


# --------------------------------------------------------------------------- grounding

# Which trace phase each op becomes depends only on whether the object is already held.
_PHASE = {("move_above", False): "approach", ("move_above", True): "move",
          ("descend_to", False): "descend", ("descend_to", True): "place",
          ("lift", True): "lift", ("lift", False): "lift",
          ("retreat", True): "retreat", ("retreat", False): "retreat"}


def ground(program: Program, observation: SceneObservation) -> GroundedProgram:
    """Program + observation -> the exact plan the UR5e path executes.

    Symbols resolve against the observation, offsets are added in metres, and the result
    is handed to `planner.validate` — so the workspace limits, the phase vocabulary, and
    the eight-phase budget are enforced by the same code the chat planner goes through.
    """
    bindings, trace, held, yaw = {}, [], False, None
    point = list(observation.points.get("gripper", [0.4, 0.0, 0.3]))
    for index, op in enumerate(program.ops):
        name = op["op"]
        if name == "detect":
            try:
                bindings[op["as"]] = observation.point(op["query"])
            except ProgramError:
                # "I looked and it is not there" is the commonest honest reason to decline,
                # so a failed lookup inside a declining program is its point, not its bug.
                if program.aborts is None:
                    raise
        elif name == "abort":
            return GroundedProgram(program, None, observation.observation_id)
        elif name == "grasp":
            trace.append({"phase": "close"}); held = True
        elif name == "release":
            trace.append({"phase": "open"}); held = False
        elif name == "on_failure":
            continue
        elif name in ("lift", "retreat"):
            point = [point[0], point[1], op["height_mm"] / 1000.0]
            trace.append({"phase": _PHASE[(name, held)], "target": list(point), "yaw": yaw})
        else:
            reference = bindings[op["ref"]]
            dx, dy = op["offset_mm"]
            x, y = reference[0] + dx / 1000.0, reference[1] + dy / 1000.0
            if name == "move_above":
                ux, uy = DIRECTIONS[op["direction"]]
                x, y = x + ux * op["standoff_mm"] / 1000.0, y + uy * op["standoff_mm"] / 1000.0
            # A commanded heading persists until the next one: the wrist does not spring back
            # between the hover and the descent it was chosen for.
            yaw = yaw if op["yaw_deg"] is None else math.radians(op["yaw_deg"])
            point = [x, y, op["height_mm"] / 1000.0]
            trace.append({"phase": _PHASE[(name, held)], "target": list(point), "yaw": yaw})

    if len(trace) > MAX_PHASES:
        raise ProgramError(f"the program grounds to {len(trace)} phases; the arm accepts at most "
                           f"{MAX_PHASES} — drop an operation")
    payload = {"intent": program.strategy, "action": "execute", "note": program.note,
               "steps": [{"object": program.object, "destination": program.destination, "trace": trace}]}
    try:
        plan = planner.validate(payload)
    except PlanError as error:
        raise ProgramError(f"the grounded waypoints were rejected: {error}")
    return GroundedProgram(program, plan.steps[0], observation.observation_id)


# --------------------------------------------------------------------------- reference programs


def canonical_program(object_name: str = "red block", destination: str = "green pad",
                      grasp_offset_mm=(0.0, 0.0), target_offset_mm=(0.0, 0.0), hover_mm: float = 240.0,
                      transit_mm: float = 300.0, grasp_mm: float = 15.0, release_mm: float | None = None,
                      lift_mm: float = 240.0, redetect: bool = False, max_attempts: int = 0,
                      grasp_yaw_deg: float | None = None, approach: str = "top", standoff_mm: float = 0.0,
                      strategy: str = "straight_pick_place", note: str = "") -> Program:
    """The correct pick-and-place, parameterised. Every diagnostic is an edit of this."""
    if release_mm is None:
        release_mm = 15.0 if destination == "green pad" else 51.0
    ops = [{"op": "detect", "query": object_name, "as": "src"},
           {"op": "detect", "query": destination, "as": "dst"},
           {"op": "move_above", "ref": "src", "offset_mm": list(grasp_offset_mm), "height_mm": hover_mm,
            "direction": approach, "standoff_mm": standoff_mm, "yaw_deg": grasp_yaw_deg},
           {"op": "descend_to", "ref": "src", "offset_mm": list(grasp_offset_mm), "height_mm": grasp_mm,
            "yaw_deg": grasp_yaw_deg},
           {"op": "grasp"},
           {"op": "lift", "height_mm": lift_mm},
           {"op": "move_above", "ref": "dst", "offset_mm": list(target_offset_mm), "height_mm": transit_mm,
            "direction": "top", "standoff_mm": 0.0},
           {"op": "descend_to", "ref": "dst", "offset_mm": list(target_offset_mm), "height_mm": release_mm},
           {"op": "release"},
           {"op": "retreat", "height_mm": lift_mm}]
    if redetect:
        ops.insert(6, {"op": "detect", "query": destination, "as": "dst"})
    if max_attempts:
        ops.append({"op": "on_failure", "policy": "redetect_regrasp", "max_attempts": max_attempts})
    return validate_program({"schema_version": SCHEMA_VERSION, "strategy": strategy,
                             "task": {"object": object_name, "destination": destination},
                             "ops": ops, "note": note})


def diagnostic_programs(object_name: str = "red block", destination: str = "green pad",
                        stale_shift_mm=(45.0, -30.0)) -> list[tuple[str, str, Program]]:
    """The diagnostic suite: seven named strategies and six named faults, as (name, kind, program).

    Scripted, not sampled, so the same named behaviours appear in every scene and a failure
    slice is interpretable — a verifier that cannot tell `bad_grasp` from `offset_grasp` has
    nowhere to hide. Reported separately from the natural Claude pool: a planted bug is not
    evidence about what Claude proposes.

    The strategies are the ones the scene suite is meant to call for. Several of them are
    indistinguishable from `correct` on a plain scene with an axis-aligned cube on a flat
    pad — an orientation-aware grasp only matters once the object is rotated, an alternate
    approach only matters once something is in the way. They are here so the pool can express
    the strategy at all, which is the half of the problem this issue owns.

    `stale_coordinates` is the one emulation: a single-step task has no earlier action to go
    stale against, so the grasp is aimed at a fixed displacement from the detected centre,
    standing in for coordinates bound before the object last moved.
    """
    strategies = [
        ("correct", canonical_program(object_name, destination, note="canonical pick and place")),
        ("redetect_regrasp",
         canonical_program(object_name, destination, redetect=True, max_attempts=1,
                           strategy="pick_place_with_recovery",
                           note="redetects the destination before placing and regrasps once on a failed lift")),
        # Blocks currently spawn axis-aligned, so aligning the jaws with the faces means yaw 0.
        # `detect` reports a centre and no orientation, so this strategy can only pin a heading,
        # not read one — until perception exposes a yaw, that is as orientation-aware as a
        # program can be. A 45 degree grasp of an axis-aligned cube pinches two corners and
        # slips about half the time, which is what the strategy exists to avoid.
        ("orientation_aware_grasp",
         canonical_program(object_name, destination, grasp_yaw_deg=0.0,
                           strategy="pick_place_oriented_grasp",
                           note="pins the wrist across the object's faces instead of leaving it to the solver")),
        ("alternate_approach",
         canonical_program(object_name, destination, approach="-x", standoff_mm=50.0,
                           strategy="pick_place_lateral_approach",
                           note="hovers to one side and descends across, rather than straight down")),
        ("offset_grasp",
         canonical_program(object_name, destination, grasp_offset_mm=(15.0, 0.0),
                           strategy="pick_place_offset_grasp",
                           note="grasps deliberately off-centre but inside the gripper's tolerance")),
        ("controlled_release",
         canonical_program(object_name, destination, release_mm=None, transit_mm=200.0,
                           strategy="pick_place_controlled_release",
                           note="carries lower and releases at the surface rather than dropping")),
        ("abort_on_uncertainty",
         validate_program({"schema_version": SCHEMA_VERSION, "strategy": "abort_on_uncertainty",
                           "task": {"object": object_name, "destination": destination},
                           "ops": [{"op": "detect", "query": object_name, "as": "src"},
                                   {"op": "abort", "reason": "the grasp is not confidently supported by "
                                                             "this observation; stopping instead of guessing"}],
                           "note": "declines to act"})),
    ]
    faults = [
        ("stale_coordinates",
         canonical_program(object_name, destination, grasp_offset_mm=stale_shift_mm,
                           strategy="pick_place_stale_binding",
                           note="grasp aimed at coordinates bound before the object last moved")),
        ("bad_grasp",
         canonical_program(object_name, destination, grasp_offset_mm=(35.0, 0.0),
                           note="grasp offset just past the gripper's lateral tolerance")),
        ("missing_lift", None),
        ("early_release", None),
        ("high_release",
         canonical_program(object_name, destination, release_mm=140.0,
                           note="releases from well above the destination")),
        ("wrong_target", None),
    ]
    out = [(name, "strategy", program) for name, program in strategies]
    for name, program in faults:
        out.append((name, "fault", program or _structural_fault(name, object_name, destination)))
    return out


def _structural_fault(name: str, object_name: str, destination: str) -> Program:
    """Faults that change the shape of the program rather than one of its numbers."""
    base = canonical_program(object_name, destination).as_json()
    ops = [dict(op) for op in base["ops"]]
    grasp_at = [op["op"] for op in ops].index("grasp")
    if name == "missing_lift":
        # No lift, and the transit runs at the lowest height the schema allows, so the block
        # never reaches the lift threshold: the arm shoves it across the table.
        ops = [op for op in ops if op["op"] != "lift"]
        for op in ops[grasp_at:]:
            if op["op"] == "move_above":
                op["height_mm"] = RANGES["hover_mm"][0]
        note = "never lifts the block clear of the table, so it is dragged to the destination"
    elif name == "early_release":
        # Drop the descent onto the destination: the gripper opens at transit height.
        descend = next(index for index, op in enumerate(ops)
                       if index > grasp_at and op["op"] == "descend_to")
        ops.pop(descend)
        note = "opens the gripper at transit height instead of descending to the destination"
    elif name == "wrong_target":
        other = next(o for o in OBJECTS if o not in (object_name, destination))
        for op in ops:
            if op["op"] == "detect" and op["query"] == destination:
                op["query"] = other
        note = f"places onto the {other} while the task asks for the {destination}"
    else:
        raise ValueError(f"no structural fault named {name!r}")
    base["ops"], base["note"] = ops, note
    base["strategy"] = f"pick_place_{name}"
    return validate_program(base)


def main():
    ap = argparse.ArgumentParser(description="Ground the canonical program on one simulator scene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--object", default="red block")
    ap.add_argument("--destination", default="green pad")
    args = ap.parse_args()

    from waddle_wm.pools import Scene   # the same seeded scene the candidate pools are built on

    scene = Scene(args.seed)
    grounded = ground(canonical_program(args.object, args.destination), scene.observation)
    print(scene.observation.text)
    print(json.dumps({"observation_id": scene.observation.observation_id,
                      "dedup_key": grounded.dedup_key(), **grounded.as_json()}, indent=2, default=float))
    scene.close()


if __name__ == "__main__":
    main()
