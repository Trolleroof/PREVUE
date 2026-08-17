"""Randomised flaw packages for the browser demo: a different way to fail on every run.

    from waddle_wm import chaos
    draw = chaos.sample(rng, detections)
    saved = chaos.apply_chaos_scene(agent.env, draw, block_xy)
    opening = chaos.build_opening_plan(instruction, detections, pad_xy, draw)

`waddle_wm.demo` ships two *frozen* flaws (a grasp 6 cm past the block, a release 22 cm short
of the pad) because a results table has to quote the same flaw every time it is regenerated.
A live demo has the opposite requirement: someone watching two runs back to back must not see
the same +6 cm twice, or the verifier looks like it memorised one number.

So this module samples the flaw instead. Every draw is one of

* a **plan flaw** — the waypoints are wrong: a random-direction grasp or place offset, a grasp
  pulled toward a neighbouring block, a grasp taken from the *wrong* block's detection, a plan
  built from a stale (pre-shift) position, or a plan built from a lied-about detection;
* a **scene challenge** — the coordinates are right and the *world* is wrong: a crowding
  neighbour, an occluder, an obstacle in the carry lane, a slippery block with a weak gripper,
  a heavy block on a bouncy contact, or a biased camera.

Magnitudes are drawn from bands chosen so the unverified baseline fails: grasp offsets exceed
`SkillAgent.rule_verdict`'s 0.028 m tolerance, place offsets exceed the 0.105 m landing radius.
`guarantee_fail` is the cheap pre-check; the server still confirms it by running the baseline.

Nothing here mutates `waddle_wm.demo.SCENARIOS` — `--replay` keeps reproducing the frozen pair.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from waddle_wm.planner import WORKSPACE, Plan, validate
from waddle_wm.sim import relling_scene as scene
from waddle_wm.sim.env import GRASP_Z, HOVER_Z, TRANSIT_Z

GRASP_TOLERANCE = 0.028        # SkillAgent.rule_verdict
PAD_RADIUS = 0.105             # the green landing zone
BLOCKS = ("red block", "blue block", "yellow block")
PARKED_YELLOW = (0.66, -0.30)  # same parking spot demo_ambiguous uses

# Relative weights inside a plan-flaw draw. Grasp errors dominate because they are the case the
# coordinates alone cannot settle once a neighbour is involved.
PLAN_FLAWS = {"random_grasp": 30, "random_place": 20, "toward_neighbor": 15,
              "wrong_object": 10, "stale_grasp": 10, "perception_lie": 15}
SCENE_ONLY_PROBABILITY = 0.30  # the rest of the time the plan itself is wrong
PERCEPTION_SCENE_PROBABILITY = 0.40   # ... and this often a perception challenge rides along

# Scene challenges that fail a *correct* plan on their own, and ones that only degrade what the
# camera can see. Only the first group may carry a scene-only draw.
FAILING_SCENES = ("neighbor_crowd", "path_obstruction", "slip_drop", "release_heavy")
PERCEPTION_SCENES = ("occlusion", "camera_bias")
SCENE_KINDS = FAILING_SCENES + PERCEPTION_SCENES


@dataclass(frozen=True)
class ChaosDraw:
    """One sampled flaw package: what the plan gets wrong, and what the world gets wrong."""

    id: str
    label: str                                   # honest one-line description, for the trace
    grasp_offset: tuple[float, float] = (0.0, 0.0)
    place_offset: tuple[float, float] = (0.0, 0.0)
    grasp_from: str = "red block"                # whose detection the grasp is built from
    target_step_index: int | None = None         # which step a compound plan was warped at
    perception_lie_xy: tuple[float, float] | None = None
    scene: dict = field(default_factory=dict)
    seed: int | None = None

    @property
    def scene_kind(self) -> str | None:
        return self.scene.get("kind")

    @property
    def grasp_magnitude(self) -> float:
        return float(np.linalg.norm(self.grasp_offset))

    @property
    def place_magnitude(self) -> float:
        return float(np.linalg.norm(self.place_offset))

    def summary(self) -> dict:
        return {"flaw_kind": self.id, "flaw": self.label, "scene_kind": self.scene_kind,
                "scene_label": self.scene.get("label"),
                "grasp_offset": [round(v, 4) for v in self.grasp_offset],
                "place_offset": [round(v, 4) for v in self.place_offset],
                "grasp_offset_cm": round(self.grasp_magnitude * 100, 1),
                "place_offset_cm": round(self.place_magnitude * 100, 1),
                "grasp_from": self.grasp_from, "target_step_index": self.target_step_index,
                "perception_lie_xy": ([round(v, 4) for v in self.perception_lie_xy]
                                      if self.perception_lie_xy else None),
                "seed": self.seed}


def offset(rng, low: float, high: float) -> np.ndarray:
    """A random direction at a random radius in `[low, high]` — the sampler `generate_suite` uses."""
    angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(low, high)
    return radius * np.array([np.cos(angle), np.sin(angle)])


def in_workspace(point, margin: float = 0.005) -> bool:
    """Is this x/y inside the box `planner.validate` accepts, with a little room to spare?"""
    return all(low + margin <= float(value) <= high - margin
               for value, (low, high) in zip(point, (WORKSPACE["x"], WORKSPACE["y"])))


def reachable_offset(rng, base, low: float, high: float, tries: int = 64) -> np.ndarray:
    """An offset in the `[low, high]` band whose endpoint the planner's workspace still accepts.

    A 22 cm release offset drawn straight off the pad lands outside the y limit about a quarter of
    the time, and `validate` would reject the plan before any arm ever saw it. Rejecting those
    directions keeps the magnitude honest — the band is unchanged, only the heading is constrained.
    """
    base = np.asarray(base, dtype=float)[:2]
    for _ in range(tries):
        vector = offset(rng, low, high)
        if in_workspace(base + vector):
            return vector
    # Every heading was blocked: aim at the middle of the workspace instead.
    towards = np.array([sum(WORKSPACE["x"]) / 2, sum(WORKSPACE["y"]) / 2]) - base
    towards = towards / max(float(np.linalg.norm(towards)), 1e-9)
    return float(rng.uniform(low, high)) * towards


def bearing(vector) -> int:
    """The compass-free angle of an offset in whole degrees, for the trace line."""
    return int(round(math.degrees(math.atan2(float(vector[1]), float(vector[0]))))) % 360


def target_block(instruction: str) -> str:
    """Which block the instruction is about, defaulting to the red one the demo is built around."""
    lowered = (instruction or "").lower()
    for name in BLOCKS:
        if name.split()[0] in lowered:
            return name
    return "red block"


def is_compound(instruction: str) -> bool:
    """Does this instruction ask for more than one pick-and-place?

    Deliberately lightweight and deliberately biased toward "yes": a false positive costs one
    Claude `propose` call, a false negative builds a single-step plan for a two-step task.
    """
    text = re.sub(r"\s+", " ", (instruction or "").lower())
    if not text:
        return False
    if re.search(r"\bthen\b|\bafter that\b|\bstack\b|\bboth\b|\beach\b|;", text):
        return True
    colours = {name for name in BLOCKS if name.split()[0] in text}
    return len(colours) > 1


def _neighbour_of(detections: dict, block: str) -> tuple[str | None, float]:
    """The nearest other detected block and its distance, or `(None, inf)`."""
    if block not in detections:
        return None, math.inf
    here = np.asarray(detections[block][:2])
    best, best_distance = None, math.inf
    for label, point in detections.items():
        if label == block or label not in BLOCKS:
            continue
        distance = float(np.linalg.norm(np.asarray(point[:2]) - here))
        if distance < best_distance:
            best, best_distance = label, distance
    return best, best_distance


def _scene_challenge(kind: str, rng, block_xy) -> dict:
    """One runtime-only scene mutation, described in the same shape `apply_chaos_scene` reads.

    Everything here is reachable on the already-compiled model — friction, masses, actuator
    limits, camera position, and the two props `relling_scene` parks below the table. Nothing
    changes a geom *size*, which would need a recompiled model and a retrained checkpoint.
    """
    x, y = float(block_xy[0]), float(block_xy[1])
    if kind == "neighbor_crowd":
        spacing = float(rng.uniform(0.075, 0.100))
        angle = float(rng.uniform(-math.pi / 3, math.pi / 3))    # broadly +x, away from the arm
        blue = (round(x + spacing * math.cos(angle), 4), round(y + spacing * math.sin(angle), 4))
        return {"kind": kind,
                "label": f"blue block crowded {spacing * 100:.1f} cm from the red one",
                "blocks": {"blue_block": [*blue, scene.BLOCK_HALF],
                           "yellow_block": [*PARKED_YELLOW, scene.BLOCK_HALF]},
                "spacing": round(spacing, 4)}
    if kind == "occlusion":
        return {"kind": kind, "label": "an occluder parked beside the block, hiding part of it",
                "occluder": [round(x + float(rng.uniform(0.045, 0.070)), 4),
                             round(y - float(rng.uniform(0.045, 0.070)), 4), 0.05]}
    if kind == "path_obstruction":
        return {"kind": kind, "label": "a post standing in the carry lane between block and pad",
                "obstacle": [round(float(rng.uniform(x - 0.02, x + 0.06)), 4),
                             round(float(rng.uniform(-0.10, -0.04)), 4), 0.125]}
    if kind == "slip_drop":
        return {"kind": kind, "label": "a slippery block and a gripper held under its usual force",
                "friction": [0.30, 0.02, 0.004], "gripper_force": float(rng.uniform(0.20, 0.30))}
    if kind == "release_heavy":
        return {"kind": kind, "label": "a block three times its usual mass on a bouncy contact",
                "mass": float(rng.uniform(0.060, 0.085)), "friction": [0.25, 0.015, 0.004],
                "solref": [0.01, 0.1]}
    if kind == "camera_bias":
        shift = offset(rng, 0.004, 0.008)
        return {"kind": kind,
                "label": f"the camera shifted {float(np.linalg.norm(shift)) * 1000:.1f} mm, "
                         "so every detection is biased at the source",
                "camera_offset": [round(float(shift[0]), 4), round(float(shift[1]), 4), 0.0]}
    raise ValueError(f"unknown scene challenge {kind!r}")


def sample(rng, detections: dict | None = None, instruction: str = "", seed: int | None = None) -> ChaosDraw:
    """Draw one flaw package for this run.

    `detections` maps label -> `point_base`, exactly what `SkillAgent.perceive` yields, and is
    what the neighbour-aware kinds need. Without it those kinds are not drawn.
    """
    detections = detections or {}
    block = target_block(instruction)
    block_xy = detections.get(block, (0.42, -0.20, 0.018))[:2]
    neighbour, neighbour_distance = _neighbour_of(detections, block)

    if rng.random() < SCENE_ONLY_PROBABILITY:
        kind = str(rng.choice(FAILING_SCENES))
        challenge = _scene_challenge(kind, rng, block_xy)
        grasp = (0.0, 0.0)
        if kind == "neighbor_crowd":
            # A crowded neighbour on its own is survivable; the demo's ambiguous case is a grasp
            # nudged toward it — inside the rules tolerance, and fatal in physics.
            nudge = float(rng.uniform(0.015, 0.025))
            towards = np.asarray(challenge["blocks"]["blue_block"][:2]) - np.asarray(block_xy)
            towards = towards / max(float(np.linalg.norm(towards)), 1e-9)
            grasp = tuple(round(float(v), 4) for v in nudge * towards)
        return ChaosDraw("scene_only", f"scene challenge: {challenge['label']}",
                         grasp_offset=grasp, grasp_from=block, scene=challenge, seed=seed)

    kinds = [name for name in PLAN_FLAWS
             if neighbour is not None or name not in ("toward_neighbor", "wrong_object")]
    weights = np.asarray([PLAN_FLAWS[name] for name in kinds], dtype=float)
    kind = str(rng.choice(kinds, p=weights / weights.sum()))

    draw = _plan_flaw(kind, rng, block, block_xy, neighbour, neighbour_distance, detections, seed)
    if not draw.scene and rng.random() < PERCEPTION_SCENE_PROBABILITY:
        challenge = _scene_challenge(str(rng.choice(PERCEPTION_SCENES)), rng, block_xy)
        draw = replace(draw, scene=challenge, label=f"{draw.label}; {challenge['label']}")
    return draw


def _plan_flaw(kind, rng, block, block_xy, neighbour, neighbour_distance, detections, seed) -> ChaosDraw:
    if kind == "random_grasp":
        vector = reachable_offset(rng, block_xy, 0.035, 0.070)
        return ChaosDraw(kind, f"grasp aimed {float(np.linalg.norm(vector)) * 100:.1f} cm from the "
                               f"{block} at {bearing(vector)}° (simulates a calibration error)",
                         grasp_offset=tuple(vector.round(4)), grasp_from=block, seed=seed)
    if kind == "random_place":
        vector = reachable_offset(rng, scene.TARGET_POS[:2], 0.12, 0.22)
        return ChaosDraw(kind, f"release point {float(np.linalg.norm(vector)) * 100:.1f} cm off the "
                               f"pad at {bearing(vector)}° (simulates a stale task frame)",
                         place_offset=tuple(vector.round(4)), grasp_from=block, seed=seed)
    if kind == "toward_neighbor":
        # The neighbour is moved first and the nudge is aimed at where it will actually be, not at
        # where it was detected — otherwise the grasp would lean toward an empty patch of table.
        challenge = _scene_challenge("neighbor_crowd", rng, block_xy)
        neighbour = "blue block"
        nudge = float(rng.uniform(0.015, 0.025))
        towards = np.asarray(challenge["blocks"]["blue_block"][:2]) - np.asarray(block_xy)
        towards = towards / max(float(np.linalg.norm(towards)), 1e-9)
        return ChaosDraw(kind, f"grasp pulled {nudge * 100:.1f} cm toward the {neighbour} — inside "
                               "the 2.8 cm coordinate tolerance, and into the neighbour in physics",
                         grasp_offset=tuple(round(float(v), 4) for v in nudge * towards),
                         grasp_from=block, scene=challenge, seed=seed)
    if kind == "wrong_object":
        return ChaosDraw(kind, f"grasp built from the {neighbour}'s detection instead of the "
                               f"{block}'s ({neighbour_distance * 100:.1f} cm away)",
                         grasp_from=neighbour, seed=seed)
    if kind == "stale_grasp":
        shift = reachable_offset(rng, block_xy, 0.040, 0.060)
        blocks = {block.replace(" ", "_"): [round(float(block_xy[0] + shift[0]), 4),
                                            round(float(block_xy[1] + shift[1]), 4), scene.BLOCK_HALF]}
        return ChaosDraw(kind, f"plan built from where the {block} was before an earlier action "
                               f"moved it {float(np.linalg.norm(shift)) * 100:.1f} cm at {bearing(shift)}°",
                         grasp_from=block,
                         scene={"kind": "prior_action", "blocks": blocks,
                                "label": f"the {block} has already been displaced "
                                         f"{float(np.linalg.norm(shift)) * 100:.1f} cm"},
                         seed=seed)
    if kind == "perception_lie":
        vector = reachable_offset(rng, block_xy, 0.035, 0.070)
        return ChaosDraw(kind, f"the detector reports the {block} {float(np.linalg.norm(vector)) * 100:.1f} cm "
                               f"from where it is, at {bearing(vector)}° — the plan is built on the lie",
                         grasp_offset=tuple(vector.round(4)), grasp_from=block,
                         perception_lie_xy=tuple(vector.round(4)), seed=seed)
    raise ValueError(f"unknown plan flaw {kind!r}")


def guarantee_fail(draw: ChaosDraw) -> bool:
    """Cheap pre-check: is this draw already outside the tolerances the demo has to beat?

    True does not promise MuJoCo will miss — only the baseline run can settle that, which is why
    `server.Session.experiment` resamples until it does. It rules out the obviously-survivable
    draw before spending a rollout on it.
    """
    if draw.grasp_magnitude > GRASP_TOLERANCE or draw.place_magnitude > PAD_RADIUS:
        return True
    if draw.id == "wrong_object":
        return True
    return draw.scene_kind in FAILING_SCENES or draw.scene_kind == "prior_action"


def worst_case(instruction: str = "", seed: int | None = None) -> ChaosDraw:
    """The fallback when repeated draws somehow survive: the largest grasp offset in the band."""
    block = target_block(instruction)
    return ChaosDraw("random_grasp", f"grasp aimed 7.0 cm past the {block} in +x "
                                     "(fallback: sampled flaws kept surviving)",
                     grasp_offset=(0.070, 0.0), grasp_from=block, seed=seed)


def apply_chaos_scene(env, draw: ChaosDraw, block_xy) -> dict:
    """Apply the sampled scene challenge and pin it across every later `env.reset`.

    `SkillAgent.observe` resets the environment on every run and threads only the red block's
    x/y through it, so a neighbour, an obstacle, or a displaced block would be wiped between the
    baseline arm and the world-model arm. Wrapping `reset` — the trick `demo_ambiguous.pin_blocks`
    uses — is what makes the two arms see the same scene.

    Returns the state `restore_chaos_scene` needs to put the model back.
    """
    spec = draw.scene or {}
    model, data = env.model, env.data
    red_geom = model.geom("red_block_geom").id
    gripper = model.actuator(scene.GRIPPER_ACTUATOR).id
    camera = model.camera("demo").id
    table_geom = model.geom("table").id
    occluder_geom = model.geom("scene_occluder").id
    saved = {"reset": env.reset,
             "friction": model.geom_friction[red_geom].copy(),
             "mass": float(model.body_mass[model.body("red_block").id]),
             "forcerange": model.actuator_forcerange[gripper].copy(),
             "solref": {name: model.geom_solref[model.geom(name).id].copy()
                        for name in ("red_block_geom", "table")},
             "cam_pos": model.cam_pos[camera].copy(),
             "occluder": model.geom_pos[occluder_geom].copy()}

    if "friction" in spec:
        model.geom_friction[red_geom] = spec["friction"]
    if "mass" in spec:
        model.body_mass[model.body("red_block").id] = spec["mass"]
    if "gripper_force" in spec:
        model.actuator_forcerange[gripper] = [-spec["gripper_force"], spec["gripper_force"]]
    if "solref" in spec:
        for geom_id in (red_geom, table_geom):
            model.geom_solref[geom_id] = spec["solref"]
    if "camera_offset" in spec:
        model.cam_pos[camera] = saved["cam_pos"] + np.asarray(spec["camera_offset"])
    model.geom_pos[occluder_geom] = spec.get("occluder") or [0.0, 0.0, -1.0]

    blocks = spec.get("blocks")
    obstacle_pos = spec.get("obstacle")
    obstacle = model.body("scene_obstacle_body").mocapid[0]
    original = env.reset

    def reset(block_xy_=None, target_xy=None, blocks_=None):
        state = original(block_xy_, target_xy, blocks_ if blocks_ is not None else blocks)
        data.mocap_pos[obstacle] = obstacle_pos or [0.0, 0.0, -1.0]
        mujoco.mj_forward(model, data)
        return state

    env.reset = reset
    env.reset(list(block_xy))
    return saved


def restore_chaos_scene(env, saved: dict) -> None:
    """Undo `apply_chaos_scene`, so the next run starts from the model the session loaded."""
    if not saved:
        return
    model = env.model
    env.reset = saved["reset"]
    model.geom_friction[model.geom("red_block_geom").id] = saved["friction"]
    model.body_mass[model.body("red_block").id] = saved["mass"]
    model.actuator_forcerange[model.actuator(scene.GRIPPER_ACTUATOR).id] = saved["forcerange"]
    for name, solref in saved["solref"].items():
        model.geom_solref[model.geom(name).id] = solref
    model.cam_pos[model.camera("demo").id] = saved["cam_pos"]
    model.geom_pos[model.geom("scene_occluder").id] = saved["occluder"]
    env.data.mocap_pos[model.body("scene_obstacle_body").mocapid[0]] = [0.0, 0.0, -1.0]
    mujoco.mj_forward(model, env.data)


def pick_place_trace(grasp, place) -> list[dict]:
    """The eight-phase shape the verifier's dynamics were trained on."""
    grasp = [round(float(grasp[0]), 4), round(float(grasp[1]), 4)]
    place = [round(float(place[0]), 4), round(float(place[1]), 4)]
    return [{"phase": "approach", "target": [*grasp, HOVER_Z]},
            {"phase": "descend", "target": [*grasp, GRASP_Z]},
            {"phase": "close"},
            {"phase": "lift", "target": [*grasp, HOVER_Z]},
            {"phase": "move", "target": [*place, TRANSIT_Z]},
            {"phase": "place", "target": [*place, GRASP_Z]},
            {"phase": "open"},
            {"phase": "retreat", "target": [*place, HOVER_Z]}]


def fit_offset(step, phases, vector) -> np.ndarray:
    """The largest version of `vector` that keeps every affected waypoint inside the workspace.

    Tried in order: the offset as drawn, its mirror image, then progressively shorter versions of
    it. A shrunk offset can end up inside the tolerances — the server notices, because it keeps
    resampling until the unchecked baseline has really missed.
    """
    vector = np.asarray(vector, dtype=float)
    targets = [entry["target"] for entry in step.trace
               if entry["phase"] in phases and "target" in entry]
    if not targets or not vector.any():
        return vector

    def fits(candidate) -> bool:
        return all(in_workspace(np.asarray(target[:2]) + candidate) for target in targets)

    for candidate in (vector, -vector):
        if fits(candidate):
            return candidate
    for step_down in range(1, 21):
        candidate = vector * (0.85 ** step_down)
        if fits(candidate):
            return candidate
    return np.zeros(2)


def build_opening_plan(instruction: str, detections: dict, pad_xy, draw: ChaosDraw,
                       steps=None) -> tuple[Plan, ChaosDraw]:
    """The flawed plan every arm starts from, and the draw updated with where the flaw landed.

    Single-step tasks are built here from the camera's own detections — the same construction
    `demo.Scenario.opening_plan` uses, with the offsets sampled rather than frozen, and no Claude
    call. Compound tasks pass `steps` from one `planner.propose`, and exactly one of those steps
    is warped instead.
    """
    if steps:
        return warp_steps(instruction, steps, draw)

    block = target_block(instruction)
    source = detections.get(draw.grasp_from, detections.get(block))
    if source is None:
        raise KeyError(f"no detection for {draw.grasp_from!r}; the scene has {sorted(detections)}")
    grasp = np.asarray(source[:2]) + np.asarray(draw.grasp_offset)
    place = np.asarray(pad_xy[:2]) + np.asarray(draw.place_offset)
    plan = validate({"intent": instruction, "action": "execute", "note": draw.label,
                     "steps": [{"object": block, "destination": "green pad",
                                "trace": pick_place_trace(grasp, place)}]})
    return plan, replace(draw, target_step_index=0)


def warp_steps(instruction: str, steps, draw: ChaosDraw) -> tuple[Plan, ChaosDraw]:
    """Take Claude's own compound plan and move one step's grasp or place by the sampled offset.

    Which step is warped is `draw.target_step_index` when it is set, so a resample can reproduce
    it; otherwise the last step is chosen, which is the one a viewer is still watching when the
    run ends. Only that step changes — every other waypoint is Claude's.
    """
    steps = list(steps)
    index = draw.target_step_index
    if index is None or not 0 <= index < len(steps):
        index = len(steps) - 1
    grasp_phases, place_phases = ("approach", "descend", "lift"), ("move", "place", "retreat")
    # Claude picks its own waypoints, so the sampled offset may point out of the workspace from
    # where it put them. `fit_offset` keeps the heading where it can and the plan valid always.
    grasp_offset = fit_offset(steps[index], grasp_phases, draw.grasp_offset)
    place_offset = fit_offset(steps[index], place_phases, draw.place_offset)
    warped = []
    for position, step in enumerate(steps):
        trace = []
        for entry in step.trace:
            entry = dict(entry)
            if position == index and "target" in entry:
                if entry["phase"] in grasp_phases and any(grasp_offset):
                    entry["target"] = [round(entry["target"][0] + grasp_offset[0], 4),
                                       round(entry["target"][1] + grasp_offset[1], 4),
                                       entry["target"][2]]
                elif entry["phase"] in place_phases and any(place_offset):
                    entry["target"] = [round(entry["target"][0] + place_offset[0], 4),
                                       round(entry["target"][1] + place_offset[1], 4),
                                       entry["target"][2]]
            trace.append(entry)
        warped.append({"object": step.object, "destination": step.destination, "trace": trace})
    draw = replace(draw, target_step_index=index,
                   grasp_offset=tuple(round(float(v), 4) for v in grasp_offset),
                   place_offset=tuple(round(float(v), 4) for v in place_offset))
    note = f"step {index + 1} of {len(steps)} was warped — {draw.label}"
    plan = validate({"intent": instruction, "action": "execute", "note": note, "steps": warped})
    return plan, draw
