"""The three selectors #18 compares, behind one scorer interface.

A selector reads a frozen pool and returns one score per candidate. It never executes, never
repairs, never rewrites, and never adds a candidate — `rank` fingerprints its input before and
after the call and refuses a run where either changed. What separates the three arms is only
what they are allowed to read:

    claude_self_rank   the task, the observation text the pool was generated against, and the
                       source programs — anonymised and shuffled, so it cannot recognise its
                       own earlier sample or read the pool order as a hint
    estimated_state    each fixed program plus the image-to-coordinate perception pipeline's
                       output (detected centres, boxes, apparent size, the task-frame pad).
                       No frames, no MuJoCo state
    visual_world_model the same programs and the same estimated coordinates, plus the raw
                       camera window and the frozen V-JEPA latent computed from it

The MuJoCo oracle in #23 is not here: it is not a selector, and nothing in this module can
see an outcome. `ScenarioContext` is the whole information boundary — the estimated-state arm
is handed a context whose `frames` is None, and `rank` checks that before it calls anything.

    uv run python -m waddle_wm.selectors --pool data/pools/diagnostic/red_block_to_green_pad/x.json
    uv run python -m waddle_wm.selectors --fit data/counterfactual/train-diagnostic.json \
        --pools data/pools --out models/estimated_state_heuristic.json

The benchmark that runs all three over locked pools and scores them against #23's outcomes is
`waddle_wm.benchmark_selectors`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from waddle_wm import plan_encoding
from waddle_wm.benchmark_record import SelectorRun, Timing, selector_choice, settings_hash
from waddle_wm.planner import MODEL, ClaudePlanner

SELECTOR_VERSION = 1

# A candidate that declines to act never executes and is never a success (see
# `benchmark_record.success_definition`). Both scoring arms read the abort() op straight off
# the program, so this is protocol, not an outcome leak — and it is the same number for every
# arm, so no selector is advantaged by it.
DECLINED_PROBABILITY = 0.02

# Nominal tabletop geometry, published in the generator's system prompt and therefore known to
# every arm. Millimetres.
BLOCK_HALF_MM = 18.0
GRASP_TARGET_MM = 15.0          # the height a centred grasp descends to
CLEARANCE_MM = 40.0             # lift height a carry needs above the block top


class SelectorError(RuntimeError):
    """A selector broke the ranking contract: it mutated, dropped, or invented a candidate."""


# --------------------------------------------------------------------------- the boundary


@dataclass
class ScenarioContext:
    """What one selector may read about one scenario. Built by the benchmark, never by a selector.

    `view` is #23's `selector_view` of the frozen pool: programs, grounded traces, the
    observation text, and the perception pipeline's detections. `frames` is the raw camera
    window and is `None` for every arm that is not the visual world model — the boundary is a
    missing field, not a promise.
    """

    view: dict
    frames: np.ndarray | None = None
    latent: object | None = None       # the frozen-encoder latent of `frames`, cached per scene

    @property
    def pool_id(self) -> str:
        return self.view["pool_id"]

    def candidates(self, prefix: list[str]) -> list[dict]:
        by_id = {candidate["candidate_id"]: candidate for candidate in self.view["candidates"]}
        missing = [cid for cid in prefix if cid not in by_id]
        if missing:
            raise SelectorError(f"{self.pool_id}: prefix names candidates that are not in the pool: {missing[:4]}")
        return [by_id[cid] for cid in prefix]

    def estimates(self) -> dict:
        """The image-derived coordinates, keyed the way the task names things.

        One dict per detected object exactly as `waddle_wm.perception` measured it, plus the
        task-frame landing pad. A block the camera could not find is simply absent, which is
        itself information the perception pipeline produced.
        """
        points = {row["label"]: row for row in self.view["scene"].get("detections") or []}
        pad = self.view["scene"].get("landing_pad") or {}
        centre = list(pad.get("centre") or [])
        if len(centre) == 2:
            centre = [*centre, 0.0]
        return {"objects": points,
                "green pad": {"label": "green pad", "point_base": centre,
                              "radius": float(pad.get("radius") or 0.0)}}


def fingerprint(payload) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=float).encode()).hexdigest()[:16]


class Selector:
    """One arm of the comparison. Subclasses implement `score`, and nothing else runs physics."""

    name = "selector"
    information_sources: tuple[str, ...] = ("observation_text",)
    needs_frames = False
    accept_threshold = 0.5

    def config(self) -> dict:
        return {"selector": self.name, "selector_version": SELECTOR_VERSION}

    @property
    def config_hash(self) -> str:
        return settings_hash(self.config())

    def prepare(self, context: ScenarioContext) -> None:
        """Per-scenario setup that is not per-candidate. Timed inside the selector window."""

    def score(self, context: ScenarioContext, prefix: list[str]) -> list[dict]:
        """One row per candidate id in `prefix`, in any order:

            {"candidate_id": ..., "score": float, "probability": float | None,
             "uncertainty": float | None, "rank": int | None}
        """
        raise NotImplementedError

    @property
    def cost_usd(self) -> float:
        return 0.0


def rank(selector: Selector, context: ScenarioContext, prefix: list[str]) -> dict:
    """Run one selector on one pool prefix and return the `SelectorRun` block #24 validates.

    The contract is enforced here rather than trusted: the selector is handed a copy so it
    cannot corrupt the pool the next arm reads, that copy is fingerprinted before and after
    the call so an attempted repair is still caught, the returned rows must be exactly the
    prefix once each, and an arm that was not granted raw frames must not have been handed any.
    """
    if context.frames is not None and not selector.needs_frames:
        raise SelectorError(f"{selector.name} was handed raw frames it is not allowed to read")
    if selector.needs_frames and context.frames is None:
        raise SelectorError(f"{selector.name} needs the raw camera window and was given none")

    guarded = ScenarioContext(deepcopy(context.view), context.frames, context.latent)
    before = fingerprint(guarded.view)
    cost_before = selector.cost_usd
    observation_ready_at = time.time()
    prepared_at = observation_ready_at
    selector.prepare(guarded)
    prepared_at = time.time()
    # An encoded observation is a property of the scene, not of the prefix: caching it back on
    # the caller's context is what stops the widest prefix paying for the encoder five times.
    context.latent = guarded.latent
    rows = selector.score(guarded, list(prefix))
    chosen_at = time.time()

    if fingerprint(guarded.view) != before or fingerprint(context.view) != fingerprint(guarded.view):
        raise SelectorError(f"{selector.name} mutated the frozen pool")
    scored = [row["candidate_id"] for row in rows]
    if sorted(scored) != sorted(prefix):
        raise SelectorError(f"{selector.name} scored {len(scored)} of {len(prefix)} candidates; a "
                            f"selector may not add, drop, or repair a candidate during ranking")
    for row in rows:
        row["score"] = float(row["score"])
        if not math.isfinite(row["score"]):
            raise SelectorError(f"{selector.name} returned a non-finite score for {row['candidate_id']}")

    timing = Timing(observation_ready_at, chosen_at,
                    {"prepare": prepared_at - observation_ready_at, "score": chosen_at - prepared_at})
    return SelectorRun(selector.name, selector.config_hash, list(selector.information_sources),
                       rows, selector_choice(rows, list(prefix)), timing.as_json(),
                       selector.cost_usd - cost_before).as_json()


# --------------------------------------------------------------------------- shared geometry


def waypoints(candidate: dict) -> dict:
    """The grounded trace read as a pick-and-place, tolerantly. Both scoring arms use this.

    A candidate is free to use the operations in any order the schema allows, so nothing here
    assumes the canonical eight phases: the grasp is the last descent before the gripper
    closes, the placement is the last motion before it opens, and anything absent is `None`.
    """
    trace = candidate.get("grounded_trace") or []
    grasp = place = approach = None
    lift_mm, carry_mm, held, last = None, None, False, None
    for entry in trace:
        phase = entry.get("phase")
        if phase == "close":
            grasp, held = last, True
        elif phase == "open":
            place = last
        elif phase in ("lift", "retreat"):
            height = 1000.0 * float(entry["target"][2])
            lift_mm = height if lift_mm is None else max(lift_mm, height)
            last = entry
        else:
            if phase == "approach" and approach is None:
                approach = entry
            if held and phase == "move":
                height = 1000.0 * float(entry["target"][2])
                carry_mm = height if carry_mm is None else max(carry_mm, height)
            last = entry
    return {"grasp": grasp, "place": place, "approach": approach,
            "lift_mm": lift_mm, "carry_mm": carry_mm,
            "phases": [entry.get("phase") for entry in trace]}


def _xy(entry) -> np.ndarray | None:
    if entry is None or "target" not in entry:
        return None
    return np.asarray(entry["target"][:2], dtype=float)


FEATURES = ("grasp_offset_mm", "grasp_height_error_mm", "place_offset_mm", "place_margin_mm",
            "release_height_mm", "lift_clearance_mm", "carry_clearance_mm", "approach_offset_mm",
            "yaw_commanded", "elongated_object", "malformed", "declined", "redetects",
            "retry_attempts", "object_undetected")


def features(candidate: dict, estimates: dict, task: dict) -> dict[str, float]:
    """One candidate, read against the perception pipeline's estimates. Millimetres throughout.

    Every number here comes from the estimated coordinates and the candidate's own grounded
    waypoints. Nothing consults the simulator, the outcome, or a rendered frame — this is
    exactly the information the estimated-state arm is defined to have.
    """
    objects, pad = estimates["objects"], estimates["green pad"]
    source = objects.get(task["object"])
    destination = pad if task["destination"] == "green pad" else objects.get(task["destination"])
    points = waypoints(candidate)
    row = {name: 0.0 for name in FEATURES}
    row["declined"] = 1.0 if candidate.get("aborts") else 0.0
    row["redetects"] = 1.0 if candidate.get("redetect_ops") else 0.0
    row["retry_attempts"] = float((candidate.get("retry") or {}).get("max_attempts") or 0)
    row["object_undetected"] = 0.0 if source else 1.0
    if row["declined"]:
        return row

    phases = points["phases"]
    row["malformed"] = 0.0 if ("close" in phases and "open" in phases
                               and phases.index("close") < phases.index("open")
                               and points["grasp"] is not None
                               and points["place"] is not None) else 1.0

    source_xy = np.asarray(source["point_base"][:2], dtype=float) if source else None
    source_z_mm = 1000.0 * float(source["point_base"][2]) if source else BLOCK_HALF_MM
    grasp_xy, place_xy, approach_xy = _xy(points["grasp"]), _xy(points["place"]), _xy(points["approach"])
    if source_xy is not None and grasp_xy is not None:
        row["grasp_offset_mm"] = float(np.linalg.norm(grasp_xy - source_xy) * 1000.0)
    if source_xy is not None and approach_xy is not None:
        row["approach_offset_mm"] = float(np.linalg.norm(approach_xy - source_xy) * 1000.0)
    if points["grasp"] is not None:
        row["grasp_height_error_mm"] = abs(1000.0 * float(points["grasp"]["target"][2]) - GRASP_TARGET_MM)
        row["yaw_commanded"] = 1.0 if points["grasp"].get("yaw") is not None else 0.0

    if destination is not None and place_xy is not None:
        destination_xy = np.asarray(destination["point_base"][:2], dtype=float)
        distance_mm = float(np.linalg.norm(place_xy - destination_xy) * 1000.0)
        radius_mm = 1000.0 * float(destination.get("radius") or 0.0)
        row["place_offset_mm"] = distance_mm
        # How far outside the destination the release is aimed; 0 while it is still inside.
        row["place_margin_mm"] = max(0.0, distance_mm - (radius_mm if radius_mm else BLOCK_HALF_MM))
        surface_mm = 0.0 if task["destination"] == "green pad" else 2 * BLOCK_HALF_MM
        row["release_height_mm"] = max(0.0, 1000.0 * float(points["place"]["target"][2]) - surface_mm)

    top_mm = source_z_mm + BLOCK_HALF_MM
    if points["lift_mm"] is not None:
        row["lift_clearance_mm"] = points["lift_mm"] - top_mm
    if points["carry_mm"] is not None:
        row["carry_clearance_mm"] = points["carry_mm"] - top_mm
    elif points["lift_mm"] is not None:
        row["carry_clearance_mm"] = row["lift_clearance_mm"]

    if source and source.get("box"):
        x0, y0, x1, y1 = source["box"]
        width, height = max(1, x1 - x0), max(1, y1 - y0)
        row["elongated_object"] = float(max(width, height) / min(width, height) - 1.0)
    return row


# The predeclared geometry rule, written as logistic weights so a fitted version is the same
# function with different numbers. Chosen before any locked result was looked at: a grasp is
# worth ~2.5 logits at 10 mm off centre, a release outside the pad is worth ~6, and a carry
# with no clearance over the block is worth ~3.
DEFAULT_WEIGHTS = {
    "bias": 5.0,
    "grasp_offset_mm": -0.25,
    "grasp_height_error_mm": -0.10,
    "place_offset_mm": -0.005,
    "place_margin_mm": -0.20,
    "release_height_mm": -0.035,
    "lift_clearance_mm": 0.0,
    "carry_clearance_mm": 0.0,
    "approach_offset_mm": -0.004,
    "yaw_commanded": -0.10,
    "elongated_object": 0.0,
    "malformed": -6.0,
    "declined": 0.0,          # handled by DECLINED_PROBABILITY before the logistic
    "redetects": 0.4,
    "retry_attempts": 0.3,
    "object_undetected": -3.0,
}
# Clearance is a floor, not a gradient: below it the carry drags the block, above it more
# height buys nothing. Applied as a hinge before the logistic.
CLEARANCE_PENALTY = -3.0


def logit(row: dict[str, float], weights: dict[str, float]) -> float:
    total = float(weights.get("bias", 0.0))
    for name in FEATURES:
        total += float(weights.get(name, 0.0)) * float(row.get(name, 0.0))
    for name in ("lift_clearance_mm", "carry_clearance_mm"):
        total += CLEARANCE_PENALTY * max(0.0, (CLEARANCE_MM - float(row.get(name, 0.0))) / CLEARANCE_MM)
    return total


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


class EstimatedStateHeuristic(Selector):
    """Geometry over the perception pipeline's coordinates. No frames, no simulator state.

    This is the arm the visual model has to beat, so it is given the perception output in
    full — centres, boxes, apparent size — and a rule that reads every part of the program a
    coordinate can reach: grasp offset, grasp height, release point and height, lift and carry
    clearance, retry policy, and whether the object was found at all.
    """

    name = "estimated_state"
    information_sources = ("observation_text", "heuristic_image_estimate")

    def __init__(self, weights: dict | None = None, fitted: dict | None = None):
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.fitted = fitted or {"source": "predeclared_default"}

    def config(self) -> dict:
        return {**super().config(), "weights": self.weights, "fitted": self.fitted,
                "clearance_penalty": CLEARANCE_PENALTY, "clearance_mm": CLEARANCE_MM,
                "declined_probability": DECLINED_PROBABILITY,
                "reads": ["detected centres", "detection boxes", "apparent size",
                          "task-frame landing pad", "candidate programs and grounded traces"]}

    def probability(self, candidate: dict, estimates: dict, task: dict) -> float:
        if candidate.get("aborts"):
            return DECLINED_PROBABILITY
        return sigmoid(logit(features(candidate, estimates, task), self.weights))

    def score(self, context: ScenarioContext, prefix: list[str]) -> list[dict]:
        estimates, task = context.estimates(), context.view["task"]
        rows = [{"candidate_id": candidate["candidate_id"],
                 "probability": self.probability(candidate, estimates, task),
                 "uncertainty": None}
                for candidate in context.candidates(prefix)]
        for row in rows:
            row["score"] = row["probability"]
        return _ranked(rows)


def _ranked(rows: list[dict]) -> list[dict]:
    """Stamp each row with its position in the selector's own ordering, best first."""
    order = sorted(rows, key=lambda row: -float(row["score"]))
    for position, row in enumerate(order):
        row["rank"] = position
    return rows


# --------------------------------------------------------------------------- Claude self-rank


SYSTEM_PROMPT = """You are ranking robot policy programs for a UR5e arm with a Robotiq 2F-85 gripper on a
tabletop. Each program was written by another model from the same camera observation you are given. You
cannot run them and you will never be told what happened. Judge which are most likely to physically
succeed at the task.

Reply with ONE JSON object and nothing else. No prose, no markdown fence.

{"ranking": ["<program label>", ...],
 "scores": {"<program label>": <probability of physical success, 0..1>, ...},
 "note": "<one or two sentences on what separated them>"}

Rules:
- `ranking` lists EVERY label exactly once, most likely to succeed first.
- `scores` gives every label an independent probability that it succeeds. Do not simply
  spread them evenly; a program you think fails should score below 0.5.
- Judge the program as written. You may not rewrite, repair, or combine programs, and you may
  not propose one of your own.
- A program that declines to act (`abort`) never places the object, but is the right answer
  when the scene genuinely does not support the task."""


def rank_prompt(task: dict, observation: str, programs: list[tuple[str, dict]]) -> str:
    lines = [f"Task instruction (from the operator):\n"
             f"pick up the {task['object']} and put it on the {task['destination']}", "",
             "Camera observation of the scene right now:", observation, "",
             f"The {len(programs)} candidate programs, in no particular order:"]
    for label, program in programs:
        lines.append(f"\n{label}:\n{json.dumps(program, indent=2)}")
    lines.append("\nRank them. Reply with the JSON object only.")
    return "\n".join(lines)


@dataclass
class ClaudeSelfRank(Selector):
    """A fresh ranking call over anonymised, shuffled, frozen candidates.

    Anonymised because a label carrying the pool index would let the ordering itself be read
    as a hint, and shuffled with a seed derived from the pool and the prefix so the same
    scene always presents the same way. It sees exactly what generation saw — the task, the
    observation text, and the programs — and it may select but not generate.
    """

    model: str = MODEL
    timeout: float = 180.0
    cache: Path | None = None
    calls: list[dict] = field(default_factory=list)

    name = "claude_self_rank"
    information_sources = ("observation_text", "claude_self_rank")

    def config(self) -> dict:
        return {**super().config(), "model": self.model,
                "system_prompt_sha1": hashlib.sha1(SYSTEM_PROMPT.encode()).hexdigest()[:12],
                "prompt_template_sha1": hashlib.sha1(rank_prompt(
                    {"object": "", "destination": ""}, "", []).encode()).hexdigest()[:12],
                "anonymised": True, "shuffled": True, "max_turns": 1, "retry_bad_reply": False,
                "declined_probability": DECLINED_PROBABILITY,
                "reads": ["task instruction", "observation text", "candidate programs"]}

    @property
    def cost_usd(self) -> float:
        return round(sum(float(call.get("cost_usd") or 0.0) for call in self.calls), 6)

    def _labels(self, context: ScenarioContext, prefix: list[str]) -> list[tuple[str, dict]]:
        """Anonymised, shuffled (label -> program), plus the label->candidate map."""
        candidates = context.candidates(prefix)
        order = list(range(len(candidates)))
        random.Random(f"{context.pool_id}:{len(prefix)}:selfrank").shuffle(order)
        return [(f"program_{position + 1:02d}", candidates[index])
                for position, index in enumerate(order)]

    def _ask(self, prompt: str, key: str) -> dict:
        """One `claude -p` call, cached by prompt so a re-run reproduces the same artifact."""
        path = self.cache / f"{key}.json" if self.cache else None
        if path is not None and path.exists():
            saved = json.loads(path.read_text())
            self.calls.append({**saved.get("call", {}), "cached": True})
            return saved
        caller = ClaudePlanner(model=self.model, timeout=self.timeout, retries=0,
                              system_prompt=SYSTEM_PROMPT)
        started = time.time()
        try:
            raw = caller.complete(prompt)
            error = None
        except (RuntimeError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as failure:
            raw, error = "", f"{type(failure).__name__}: {failure}"
        call = {**(caller.calls[-1] if caller.calls else {"model": self.model}),
                "seconds": round(time.time() - started, 3)}
        saved = {"raw": raw, "error": error, "call": call}
        self.calls.append(call)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(saved, indent=2))
        return saved

    def score(self, context: ScenarioContext, prefix: list[str]) -> list[dict]:
        labelled = self._labels(context, prefix)
        prompt = rank_prompt(context.view["task"], context.view["scene"]["observation"],
                             [(label, candidate["program"]) for label, candidate in labelled])
        key = f"{context.pool_id}-n{len(prefix):02d}-{fingerprint({'prompt': prompt, 'config': self.config()})}"
        reply = self._ask(prompt, key)
        ranking, scores = parse_ranking(reply["raw"], [label for label, _ in labelled])

        rows = []
        for label, candidate in labelled:
            # A label Claude never mentioned is unranked, not silently last-but-tied: it gets
            # the floor probability and sorts behind everything it did rank.
            probability = scores.get(label)
            position = ranking.index(label) if label in ranking else None
            rows.append({"candidate_id": candidate["candidate_id"],
                         "score": 0.0 if probability is None else float(probability),
                         "probability": None if probability is None else float(probability),
                         "uncertainty": None, "rank": position})
        return rows


def parse_ranking(raw: str, labels: list[str]) -> tuple[list[str], dict[str, float]]:
    """Claude's reply -> (ranking, per-label probability). A broken reply ranks nothing."""
    stripped = (raw or "").strip()
    for fence in ("```json", "```"):
        if stripped.startswith(fence):
            stripped = stripped[len(fence):].removesuffix("```").strip()
            break
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        return [], {}
    try:
        payload = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError:
        return [], {}
    known = set(labels)
    ranking, seen = [], set()
    for label in payload.get("ranking") or []:
        if label in known and label not in seen:
            ranking.append(label)
            seen.add(label)
    scores = {}
    for label, value in (payload.get("scores") or {}).items():
        if label in known and isinstance(value, (int, float)) and not isinstance(value, bool):
            scores[label] = max(0.0, min(1.0, float(value)))
    # A ranking without scores is still a ranking: read the order as a descending score so the
    # reply is not thrown away over a missing field.
    for position, label in enumerate(ranking):
        scores.setdefault(label, round(1.0 - position / max(1, len(ranking)), 6))
    return ranking, scores


# --------------------------------------------------------------------------- visual world model


class VisualWorldModel(Selector):
    """The estimated coordinates the heuristic gets, plus the raw camera window.

    The checkpoint is the multi-block state world model: a frozen V-JEPA latent of the
    observation window, the estimated block coordinates, the candidate's grasp and place
    offsets, and the task, in — predicted terminal state and a success probability out, over
    an ensemble whose spread is the reported uncertainty.

    The plan half of that input is `waddle_wm.plan_encoding`, versioned: v2 carries the grasp
    and approach wrist headings, so two candidates differing only in `yaw_deg` reach the
    network as different vectors. A checkpoint that predates the version, or one whose yaw
    dimensions were constant while it was fitted, cannot make that distinction — it is
    refused here by name rather than quietly ranking distinct orientations as identical.
    `allow_orientation_blind=True` runs one anyway, and the arm then records
    `orientation_blind` in its config so the limitation travels with the result.
    """

    name = "visual_world_model"
    information_sources = ("observation_text", "heuristic_image_estimate",
                           "visual_model_frames", "visual_model_latents")
    needs_frames = True
    # Two guards on the checkpoint's own normalisation, applied to every candidate equally.
    # A feature whose training standard deviation collapsed to the clamp floor was *constant*
    # while the model was fitted — the grasp height above the block is, because every recorded
    # episode descended to the same height over a block whose centre was read from the
    # simulator. Dividing a live estimate's millimetre of perception noise by 1e-6 hands the
    # network a five-thousand-sigma input and it saturates to p=0 for every candidate, which
    # is not a judgement, it is an overflow. Degenerate dimensions are held at their training
    # constant and the rest are clamped to the range the fit actually covered.
    DEGENERATE_STD = 1e-5
    FEATURE_CLAMP = 5.0

    def __init__(self, checkpoint: Path, encoder: Path = Path("models/vjepa2-vitl-fpc64-256"),
                 device=None, allow_orientation_blind: bool = False):
        import torch
        from torch import nn

        from waddle_wm.train_multiblock_world_model import StateWorldModel

        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("model_type") != "multiblock_state":
            raise SelectorError(f"{checkpoint} is a {saved.get('model_type')!r} checkpoint; the visual "
                                f"selector needs a multiblock_state world model")
        self.plan_encoding = plan_encoding.declared(saved)
        if len(self.plan_encoding["fields"]) != int(saved["plan_dim"]):
            raise SelectorError(f"{checkpoint} declares a {len(self.plan_encoding['fields'])}-field plan "
                                f"encoding but was fitted with plan_dim={saved['plan_dim']}")
        self.orientation_blind = plan_encoding.orientation_blind(self.plan_encoding)
        plan_encoding.require_orientation_aware(f"{checkpoint}", self.plan_encoding,
                                                allow_orientation_blind, error=SelectorError)
        self.torch = torch
        self.checkpoint, self.encoder_path = Path(checkpoint), Path(encoder)
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.manifest = saved["manifest"]
        self.block_names = tuple(self.manifest["block_names"])
        self.threshold = float(saved.get("decision_threshold", 0.5))
        self.norm = {key: value.to(self.device) for key, value in saved["normalization"].items()}
        self.members = nn.ModuleList([StateWorldModel(saved["context_dim"], saved["plan_dim"])
                                      for _ in range(saved["member_count"])])
        self.members.load_state_dict(saved["members"])
        self.members.to(self.device).eval()
        self.metrics = saved.get("metrics", {})
        self._encoder = None

    @property
    def accept_threshold(self) -> float:
        return self.threshold

    def config(self) -> dict:
        return {**super().config(), "checkpoint": str(self.checkpoint),
                "checkpoint_sha1": _file_sha1(self.checkpoint),
                "encoder": str(self.encoder_path), "members": len(self.members),
                "window_frames": self.manifest["window_frames"], "fps": self.manifest["fps"],
                "accept_threshold": self.threshold,
                "plan_encoding": self.plan_encoding, "orientation_blind": self.orientation_blind,
                "degenerate_std": self.DEGENERATE_STD, "feature_clamp": self.FEATURE_CLAMP,
                "degenerate_plan_dims": self._degenerate("plan_std"),
                "degenerate_state_dims": self._degenerate("state_std"),
                "declined_probability": DECLINED_PROBABILITY,
                "fitted_on": {"data": self.manifest.get("name") or self.manifest.get("episodes"),
                              "splits": "train/val of the world-model dataset, never the pool split"},
                "reads": ["raw camera window", "frozen V-JEPA latent", "detected centres",
                          "task-frame landing pad", "candidate programs and grounded traces"]}

    # ----------------------------------------------------------------- encoding

    def encode(self, frames) -> object:
        """Raw camera window -> the frozen encoder's latent, through the training codec path."""
        from waddle_wm.verifier import through_codec

        window = self.manifest["window_frames"]
        frames = np.asarray(frames)
        if len(frames) < window:
            frames = np.concatenate([frames, np.repeat(frames[-1:], window - len(frames), axis=0)])
        frames = through_codec(frames[:window], self.manifest["fps"])
        if self._encoder is None:
            from transformers import AutoModel, AutoVideoProcessor
            self._encoder = (AutoVideoProcessor.from_pretrained(self.encoder_path, local_files_only=True),
                             AutoModel.from_pretrained(self.encoder_path, local_files_only=True)
                             .to(self.device).eval())
        processor, encoder = self._encoder
        with self.torch.inference_mode():
            pixels = processor(list(frames), return_tensors="pt")["pixel_values_videos"].to(self.device)
            latent = encoder(pixel_values_videos=pixels).last_hidden_state.mean(dim=1).float()
        return self._normalise(latent, "context")

    def prepare(self, context: ScenarioContext) -> None:
        if context.latent is None:
            context.latent = self.encode(context.frames)
        self._latent = context.latent

    # ----------------------------------------------------------------- scoring

    def _degenerate(self, key: str) -> list[int]:
        return [index for index, value in enumerate(self.norm[key].flatten().tolist())
                if value <= self.DEGENERATE_STD]

    def _normalise(self, values, key: str):
        """Standardise against the checkpoint, holding degenerate dimensions at their constant."""
        std = self.norm[f"{key}_std"]
        scaled = (values - self.norm[f"{key}_mean"]) / std
        scaled = self.torch.where(std <= self.DEGENERATE_STD,
                                  self.torch.zeros_like(scaled), scaled)
        return scaled.clamp(-self.FEATURE_CLAMP, self.FEATURE_CLAMP)

    def _state(self, estimates: dict):
        """Estimated block coordinates in the checkpoint's block order, as the model's state."""
        objects = estimates["objects"]
        values = []
        for name in self.block_names:
            row = objects.get(name.replace("_", " "))
            values.extend(row["point_base"][:3] if row else [0.5, 0.0, BLOCK_HALF_MM / 1000.0])
        return self.torch.tensor(values, dtype=self.torch.float32, device=self.device).unsqueeze(0)

    def plan_row(self, candidate: dict, estimates: dict, task: dict, state) -> np.ndarray | None:
        """The candidate's action in the checkpoint's plan encoding, or None if it has no trace.

        Grasp and release as offsets from the estimated coordinates they aim at, plus — from
        encoding v2 on — the wrist heading the program pinned for the descent and for the
        approach before it. The headings come off the candidate's own grounded trace, so two
        candidates that differ only in `yaw_deg` differ here too.
        """
        points = waypoints(candidate)
        if points["grasp"] is None or points["place"] is None:
            return None
        source = self.block_names.index(task["object"].replace(" ", "_"))
        source_xyz = state[0, source * 3:source * 3 + 3].detach().cpu().numpy()
        if task["destination"] == "green pad":
            destination_xyz = np.asarray(estimates["green pad"]["point_base"], dtype=float)
        else:
            index = self.block_names.index(task["destination"].replace(" ", "_"))
            destination_xyz = state[0, index * 3:index * 3 + 3].detach().cpu().numpy()
        approach = points["approach"] or {}
        return plan_encoding.plan_vector(points["grasp"]["target"], points["place"]["target"],
                                         source_xyz, destination_xyz,
                                         points["grasp"].get("yaw"), approach.get("yaw"),
                                         self.plan_encoding["version"])

    def _plan(self, candidate: dict, estimates: dict, task: dict, state):
        row = self.plan_row(candidate, estimates, task, state)
        if row is None:
            return None
        return self.torch.from_numpy(row).to(self.device).unsqueeze(0)

    def score(self, context: ScenarioContext, prefix: list[str]) -> list[dict]:
        from waddle_wm.train_multiblock_world_model import task_features

        estimates, task = context.estimates(), context.view["task"]
        raw_state = self._state(estimates)
        state = self._normalise(raw_state, "state")
        task_row = task_features([task["object"].replace(" ", "_")],
                                 [task["destination"].replace(" ", "_")], self.block_names, self.device)
        rows = []
        for candidate in context.candidates(prefix):
            plan = None if candidate.get("aborts") else self._plan(candidate, estimates, task, raw_state)
            if plan is None:
                # Declines and malformed traces are floored by the shared constant, so the two
                # scoring arms treat them identically and the comparison is about geometry.
                rows.append({"candidate_id": candidate["candidate_id"], "score": DECLINED_PROBABILITY,
                             "probability": DECLINED_PROBABILITY, "uncertainty": 0.0})
                continue
            plan = self._normalise(plan, "plan")
            with self.torch.inference_mode():
                logits = self.torch.stack([member(self._latent, state, plan, task_row)[1]
                                           for member in self.members])
            scores = logits[..., 1].sigmoid()
            probability, uncertainty = float(scores.mean()), float(scores.std())
            rows.append({"candidate_id": candidate["candidate_id"], "score": probability,
                         "probability": probability, "uncertainty": uncertainty})
        return _ranked(rows)


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


# --------------------------------------------------------------------------- calibration


def fit_heuristic(artifacts: list[dict], views: dict) -> dict:
    """Refit the geometry rule's weights on executed outcomes from a non-test split.

    The rule's *form* is fixed; only its coefficients move, and they may only move on train or
    calibration scenes. Refusing a test artifact here is the whole point of the function: a
    heuristic tuned on the locked split would make the comparison meaningless in the direction
    that flatters the learned arm.
    """
    from sklearn.linear_model import LogisticRegression

    rows, labels, splits, pools = [], [], set(), set()
    for artifact in artifacts:
        split = (artifact.get("metadata", {}).get("dataset") or {}).get("split")
        splits.add(split)
        if split not in ("train", "calibration"):
            raise SelectorError(f"refusing to fit the heuristic on the {split!r} split; "
                                f"learned components are fitted on train/calibration only")
        for scene in artifact.get("scenes") or []:
            if scene["prefix"] != max(s["prefix"] for s in artifact["scenes"]
                                      if s["scenario_id"] == scene["scenario_id"]
                                      and s["physics_seed"] == scene["physics_seed"]):
                continue                      # one row per candidate, from the widest prefix
            view = views.get(scene["pool_id"])
            if view is None:
                continue
            pools.add(scene["pool_id"])
            context = ScenarioContext(view)
            estimates, task = context.estimates(), view["task"]
            for candidate in context.candidates(scene["pool_prefix"]):
                if candidate.get("aborts"):
                    continue                  # scored by the shared constant, not by the rule
                rows.append([features(candidate, estimates, task)[name] for name in FEATURES])
                labels.append(bool(scene["counterfactual"][candidate["candidate_id"]]["success"]))

    if len(set(labels)) < 2:
        raise SelectorError("the fitting set has only one outcome class; nothing to calibrate")
    model = LogisticRegression(max_iter=2000, C=1.0).fit(np.asarray(rows, dtype=float), labels)
    weights = {"bias": float(model.intercept_[0])}
    weights.update({name: float(value) for name, value in zip(FEATURES, model.coef_[0])})
    return {"weights": weights,
            "fitted": {"source": "logistic_regression", "splits": sorted(splits),
                       "pools": len(pools), "candidates": len(rows),
                       "success_rate": round(float(np.mean(labels)), 4),
                       "note": "the hinge on lift and carry clearance is fixed, not fitted"}}


def load_heuristic(path: Path | None) -> EstimatedStateHeuristic:
    if path is None:
        return EstimatedStateHeuristic()
    saved = json.loads(Path(path).read_text())
    return EstimatedStateHeuristic(saved["weights"], saved["fitted"])


# --------------------------------------------------------------------------- cli


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", type=Path, help="score one cached pool with the offline arms and print it")
    ap.add_argument("--weights", type=Path, help="fitted heuristic weights")
    ap.add_argument("--fit", type=Path, nargs="+", help="train/calibration counterfactual artifacts")
    ap.add_argument("--views", type=Path, nargs="+", help="matching -views.json files for --fit")
    ap.add_argument("--out", type=Path, default=Path("models/estimated_state_heuristic.json"))
    args = ap.parse_args()

    if args.fit:
        views = {}
        for path in (args.views or [p.with_name(f"{p.stem}-views.json") for p in args.fit]):
            views.update(json.loads(Path(path).read_text()))
        fitted = fit_heuristic([json.loads(path.read_text()) for path in args.fit], views)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(fitted, indent=2) + "\n")
        print(json.dumps(fitted, indent=2))
        print(f"wrote {args.out}")
        return

    if not args.pool:
        raise SystemExit("nothing to do: pass --pool to score a cached pool, or --fit to calibrate")

    from waddle_wm.counterfactual import selector_view

    pool = json.loads(args.pool.read_text())
    context = ScenarioContext(selector_view(pool))
    prefix = [candidate["candidate_id"] for candidate in context.view["candidates"]]
    block = rank(load_heuristic(args.weights), context, prefix)
    named = {candidate["candidate_id"]: candidate.get("diagnostic") or candidate.get("strategy")
             for candidate in context.view["candidates"]}
    for row in sorted(block["scores"], key=lambda row: -row["score"]):
        print(f"  {row['score']:.3f}  {named.get(row['candidate_id'])}")
    print(json.dumps({"chosen": block["chosen"], "seconds": block["timing"]["selector_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
