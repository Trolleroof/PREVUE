"""The benchmark artifact: what a locked run must record, and the checks that refuse a bad one.

A selector result is only evidence if the run it came from is pinned down. The same number
moves when the pool changes, when the observation changes, when the success threshold moves,
when the oracle tie-break flips, or when Claude and MuJoCo time leaks into "selector latency".
So the artifact records all of it, and `check_run` refuses a run that cannot be trusted:

    metadata   protocol, git SHA, split and seeds, pool ids and order, generator settings,
               perception and camera settings, success and oracle definitions, selector
               configuration and each selector's information boundary
    scenes     one entry per (scenario, physics seed, pool prefix): the pool prefix, the
               counterfactual outcome of every candidate in it (#23), the oracle candidate
               under the locked ordering, and one block per selector with its per-candidate
               scores, its choice, its score margin, its tie-break, and its latency
    excluded   every scene that was attempted and left out, with the reason

Three things are locked here and nowhere else, because they decide the answer and must be
fixed before anyone looks at a selector's results:

* `oracle_best` — success, then fewer failed attempts, then lower target error, then shorter
  execution time, then pool index. Quantised by `ORACLE_TOLERANCE` so simulator jitter cannot
  promote a candidate.
* `selector_choice` — highest score, ties broken by pool index. Never by outcome.
* the timing boundary — selector latency runs from the observation being available to the
  final candidate being chosen. Claude generation and MuJoCo execution are recorded, but in
  their own fields, and `check_run` fails if either is inside the selector window.

    uv run python -m waddle_wm.benchmark_record --validate results/programs/*.json
    uv run python -m waddle_wm.test_benchmark_record       # the negative fixtures

The counterfactual outcome block is written by #23 and read here; this module never executes
anything. `OUTCOME_FIELDS` is the contract between them.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from waddle_wm.pools import PREFIXES, PROTOCOL_VERSION, SPLITS, settings_hash
from waddle_wm.program import SCHEMA_VERSION
from waddle_wm.sim.env import LIFT_THRESHOLD, TARGET_RADIUS

ARTIFACT_VERSION = 1

# Predeclared before the locked run. A report may show more, but these three are the outcomes
# the benchmark is answering, and nothing may be promoted to primary after the fact.
PRIMARY_OUTCOMES = ("selected_success", "selection_efficiency", "oracle_gap")

# What #23 records for one candidate executed from the restored snapshot. Every field is
# required: a partially recorded execution is a hole in the oracle, not a missing nicety.
OUTCOME_FIELDS = ("success", "failure_mode", "max_lift_mm", "final_target_error_mm",
                  "failed_attempts", "timed_out", "error", "execution_seconds", "execution_order")

# The oracle ordering, locked. Read as a sort key: every entry is "smaller is better".
ORACLE_ORDER = ("not success", "failed_attempts", "final_target_error_mm", "execution_seconds",
                "pool index")
# Bucket width applied before comparison, so a re-run whose physics differ in the last digit
# does not reorder the answer key. Two candidates in one bucket tie and fall to the next key.
# Bucketing, not a pairwise tolerance: "within 0.5 mm of each other" is not transitive and
# would make the ordering depend on which pair was compared first.
ORACLE_TOLERANCE = {"final_target_error_mm": 0.5, "execution_seconds": 0.05}
# Two selector scores closer than this are a tie; the earlier pool index wins. Outcome-blind.
SCORE_TIE_EPS = 1e-6

# Selector latency is measured across this boundary and no wider.
TIMING_BOUNDARY = ("observation_ready_at", "chosen_at")
# Recorded per scene, outside every selector block. Neither may appear inside one.
EXCLUDED_FROM_SELECTOR = ("claude_generation_seconds", "mujoco_execution_seconds")

# Keys that carry simulator state or the answer key. None of them may appear anywhere under a
# selector's block: a selector that can read them is not being measured on what it can see.
# A selector's own *prediction* of a failure goes in `likely_failure`, matching `verifier.py`;
# `failure_mode` is the executed outcome and is reserved for the counterfactual record.
LEAKED_FIELDS = ("hidden_truth", "state_before", "state_after", "block_pos", "target_pos",
                 "qpos", "qvel", "max_block_z", "target_distance", "counterfactual", "oracle",
                 "oracle_gap", "actual_success", "failure_mode", "pool_has_success")

# What a selector is allowed to have seen. `hidden_oracle` and `mujoco_state` are listed so a
# run that declares them fails loudly rather than being quietly comparable to an honest one.
INFORMATION_SOURCES = ("observation_text", "claude_self_rank", "heuristic_image_estimate",
                       "visual_model_frames", "visual_model_latents")
FORBIDDEN_SOURCES = ("hidden_oracle", "mujoco_state", "counterfactual_outcome")

REQUIRED_METADATA = ("protocol_version", "program_schema_version", "git_sha", "git_dirty",
                     "dataset", "pools", "generator", "perception", "physics",
                     "success_definition", "oracle_definition", "selectors", "primary_outcomes")


def success_definition() -> dict:
    """The success test the executor actually applies, copied into the artifact verbatim."""
    return {"lift_threshold_m": LIFT_THRESHOLD, "pad_radius_m": TARGET_RADIUS,
            "rule": "max lift above threshold, final target distance within the radius, "
                    "and for a block destination the object resting on top of it",
            "aborting_candidate": "an abort() candidate never executes and is never a success"}


def oracle_definition() -> dict:
    return {"order": list(ORACLE_ORDER), "tolerance": dict(ORACLE_TOLERANCE),
            "final_tie_break": "pool index (earlier sample wins)",
            "note": "hidden offline answer key computed after executing every candidate from "
                    "the restored snapshot; not a deployable verifier"}


def definition_hash() -> str:
    """One hash over everything that decides what 'success' and 'best' mean."""
    return settings_hash({"success": success_definition(), "oracle": oracle_definition(),
                          "primary_outcomes": list(PRIMARY_OUTCOMES),
                          "artifact_version": ARTIFACT_VERSION})


# --------------------------------------------------------------------------- oracle


def _quantise(value: float, step: float) -> float:
    return math.floor(float(value) / step + 1e-9) * step


def oracle_key(outcome: dict, index: int) -> tuple:
    """The locked ordering, as a sort key. Smaller is better; see `ORACLE_ORDER`."""
    return (0 if outcome["success"] else 1,
            int(outcome["failed_attempts"]),
            _quantise(outcome["final_target_error_mm"], ORACLE_TOLERANCE["final_target_error_mm"]),
            _quantise(outcome["execution_seconds"], ORACLE_TOLERANCE["execution_seconds"]),
            index)


def oracle_ranking(prefix: list[str], outcomes: dict) -> list[str]:
    """Every candidate in the prefix, best first, under the locked ordering."""
    return sorted(prefix, key=lambda cid: oracle_key(outcomes[cid], prefix.index(cid)))


def oracle_best(prefix: list[str], outcomes: dict) -> dict:
    """The answer key for one pool prefix: the best candidate and what decided it.

    `decided_by` says which key separated the winner from the runner-up, so a report can tell
    a decisive oracle from one that came down to a 50 ms difference in execution time.
    """
    ranking = oracle_ranking(prefix, outcomes)
    best = ranking[0]
    keys = {cid: oracle_key(outcomes[cid], prefix.index(cid)) for cid in ranking}
    tied = [cid for cid in ranking if keys[cid][:-1] == keys[best][:-1]]
    if len(ranking) == 1:
        decided_by = "only candidate"
    else:
        runner_up = keys[ranking[1]]
        decided_by = next((name for name, mine, theirs in zip(ORACLE_ORDER, keys[best], runner_up)
                           if mine != theirs), ORACLE_ORDER[-1])
    return {"candidate_id": best, "ranking": ranking, "key": list(keys[best]),
            "decided_by": decided_by, "tied_with": [cid for cid in tied if cid != best],
            "pool_has_success": any(outcomes[cid]["success"] for cid in prefix)}


def selector_choice(scores: list[dict], prefix: list[str]) -> dict:
    """Argmax over scores, ties broken by pool index. Outcome-blind, by construction.

    `scores` is the selector's own output: one `{candidate_id, score, ...}` per candidate in
    the prefix. The margin is to the runner-up, and is what a report needs to say whether a
    selector was confident or was one float away from a different answer.
    """
    ordered = sorted(scores, key=lambda row: (-float(row["score"]), prefix.index(row["candidate_id"])))
    best = ordered[0]
    tied = [row["candidate_id"] for row in ordered[1:]
            if abs(float(row["score"]) - float(best["score"])) <= SCORE_TIE_EPS]
    margin = (float(best["score"]) - float(ordered[1]["score"])) if len(ordered) > 1 else float("inf")
    return {"candidate_id": best["candidate_id"], "score": float(best["score"]),
            "score_margin": round(margin, 6) if margin != float("inf") else None,
            "tie_break": "pool_index" if tied else "unique", "tied_with": tied}


# --------------------------------------------------------------------------- records


@dataclass
class Timing:
    """One selector's latency across the locked boundary, plus its component breakdown.

    `observation_ready_at` is when the observation the selector consumes exists; `chosen_at`
    is when the final candidate is picked. Candidate generation and MuJoCo execution sit
    outside this window and are recorded per scene, not here.
    """

    observation_ready_at: float
    chosen_at: float
    components: dict[str, float] = field(default_factory=dict)

    @property
    def selector_seconds(self) -> float:
        return round(self.chosen_at - self.observation_ready_at, 6)

    def as_json(self) -> dict:
        return {"observation_ready_at": self.observation_ready_at, "chosen_at": self.chosen_at,
                "selector_seconds": self.selector_seconds,
                "components": {k: round(float(v), 6) for k, v in self.components.items()}}


@dataclass
class SelectorRun:
    """What one selector did on one pool prefix. Everything here is selector-visible."""

    selector: str
    config_hash: str
    information_sources: list[str]
    scores: list[dict]          # {candidate_id, score, probability, uncertainty, rank}
    chosen: dict                # from `selector_choice`
    timing: dict                # from `Timing.as_json`
    cost_usd: float = 0.0

    def as_json(self) -> dict:
        return {"selector": self.selector, "config_hash": self.config_hash,
                "information_sources": list(self.information_sources),
                "scores": self.scores, "chosen": self.chosen, "timing": self.timing,
                "cost_usd": round(float(self.cost_usd), 6)}


@dataclass
class SceneRun:
    """One (scenario, physics seed, pool prefix) cell: the unit every paired metric is paired on."""

    scenario_id: str
    split: str
    scene_seed: int
    physics_seed: int
    pool_id: str
    pool_kind: str
    observation_id: str
    prefix: int
    pool_prefix: list[str]
    counterfactual: dict           # candidate_id -> the #23 outcome record
    selectors: dict                # name -> SelectorRun.as_json()
    execution_budget: dict
    claude_generation_seconds: float = 0.0
    mujoco_execution_seconds: float = 0.0

    def as_json(self) -> dict:
        oracle = oracle_best(self.pool_prefix, self.counterfactual)
        return {"scenario_id": self.scenario_id, "split": self.split,
                "scene_seed": self.scene_seed, "physics_seed": self.physics_seed,
                "pool_id": self.pool_id, "pool_kind": self.pool_kind,
                "observation_id": self.observation_id, "prefix": self.prefix,
                "pool_prefix": list(self.pool_prefix), "counterfactual": self.counterfactual,
                "pool_has_success": oracle["pool_has_success"], "oracle": oracle,
                "execution_budget": self.execution_budget,
                "claude_generation_seconds": round(float(self.claude_generation_seconds), 3),
                "mujoco_execution_seconds": round(float(self.mujoco_execution_seconds), 3),
                "selectors": self.selectors}


def run_metadata(split: str, scene_seeds, physics_seeds, pools: dict, generator: dict,
                 perception: dict, physics: dict, selectors: dict, git_sha: str,
                 git_dirty: bool, created_at: str) -> dict:
    """The immutable header. A saved artifact must specify the run without shell history."""
    return {"protocol_version": PROTOCOL_VERSION, "program_schema_version": SCHEMA_VERSION,
            "artifact_version": ARTIFACT_VERSION, "git_sha": git_sha, "git_dirty": git_dirty,
            "created_at": created_at,
            "dataset": {"split": split, "scene_seeds": sorted(scene_seeds),
                        "physics_seeds": sorted(physics_seeds),
                        "declared_ranges": {name: [min(seeds), max(seeds)] for name, seeds in SPLITS.items()}},
            "pools": pools, "generator": generator, "perception": perception, "physics": physics,
            "success_definition": success_definition(), "oracle_definition": oracle_definition(),
            "definition_hash": definition_hash(), "selectors": selectors,
            "primary_outcomes": list(PRIMARY_OUTCOMES)}


# --------------------------------------------------------------------------- integrity


def _leaked_keys(node, path="") -> list[str]:
    """Every `LEAKED_FIELDS` key anywhere under a selector's block."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in LEAKED_FIELDS:
                found.append(f"{path}.{key}".lstrip("."))
            found.extend(_leaked_keys(value, f"{path}.{key}".lstrip(".")))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_leaked_keys(value, f"{path}[{index}]"))
    return found


def check_timing(scene: dict, name: str, block: dict) -> list[str]:
    """The timing boundary: complete, ordered, and free of generation and execution time."""
    problems, where = [], f"{scene['scenario_id']}/{name}"
    timing = block.get("timing")
    if not timing:
        return [f"{where}: no selector timing recorded"]
    for key in (*TIMING_BOUNDARY, "selector_seconds", "components"):
        if timing.get(key) is None:
            problems.append(f"{where}: timing is missing {key}")
    if problems:
        return problems
    span = timing["chosen_at"] - timing["observation_ready_at"]
    if span < 0:
        problems.append(f"{where}: the candidate was chosen before the observation was available")
    if abs(span - timing["selector_seconds"]) > 1e-3:
        problems.append(f"{where}: selector_seconds {timing['selector_seconds']} does not match "
                        f"the {span:.6f}s boundary it claims to measure")
    components = timing["components"]
    if not components:
        problems.append(f"{where}: no component timings; the breakdown must be retained")
    if any(float(value) < 0 for value in components.values()):
        problems.append(f"{where}: negative component timing")
    if sum(float(value) for value in components.values()) > timing["selector_seconds"] + 1e-3:
        problems.append(f"{where}: component timings sum past the selector window")
    for key in EXCLUDED_FROM_SELECTOR:
        if key in components or key in timing:
            problems.append(f"{where}: {key} is inside the selector window; keep Claude and "
                            f"MuJoCo timing separate")
    return problems


def check_scene(scene: dict, metadata: dict) -> list[str]:
    problems, where = [], scene.get("scenario_id", "?")
    prefix = scene.get("pool_prefix") or []
    outcomes = scene.get("counterfactual") or {}

    if len(set(prefix)) != len(prefix):
        problems.append(f"{where}: the pool prefix repeats a candidate")
    if len(prefix) != scene.get("prefix"):
        problems.append(f"{where}: prefix {scene.get('prefix')} holds {len(prefix)} candidates")
    if scene.get("prefix") not in PREFIXES:
        problems.append(f"{where}: prefix {scene.get('prefix')} is not one of {list(PREFIXES)}")
    for candidate_id in prefix:
        outcome = outcomes.get(candidate_id)
        if outcome is None:
            problems.append(f"{where}: {candidate_id} has no counterfactual execution record")
            continue
        missing = [key for key in OUTCOME_FIELDS if key not in outcome]
        if missing:
            problems.append(f"{where}: {candidate_id} outcome is missing {', '.join(missing)}")
    for candidate_id in outcomes:
        if candidate_id not in prefix:
            problems.append(f"{where}: {candidate_id} was executed but is not in the pool prefix")
    orders = [outcome.get("execution_order") for outcome in outcomes.values()]
    if len(set(orders)) != len(orders):
        problems.append(f"{where}: two candidates share an execution order")

    if not problems:
        oracle = oracle_best(prefix, outcomes)
        recorded = scene.get("oracle") or {}
        if recorded.get("candidate_id") != oracle["candidate_id"]:
            problems.append(f"{where}: recorded oracle {recorded.get('candidate_id')} is not the "
                            f"locked ordering's answer ({oracle['candidate_id']})")
        if scene.get("pool_has_success") != oracle["pool_has_success"]:
            problems.append(f"{where}: pool_has_success disagrees with the executions")

    declared = metadata.get("selectors") or {}
    if set(scene.get("selectors") or {}) != set(declared):
        problems.append(f"{where}: selectors {sorted(scene.get('selectors') or {})} do not match the "
                        f"declared {sorted(declared)}; a paired comparison needs every arm")
    for name, block in (scene.get("selectors") or {}).items():
        # Exactly once each: a missing score is a candidate the selector never judged, and a
        # repeated one is a candidate it got two chances at.
        scored = [row["candidate_id"] for row in block.get("scores", [])]
        if sorted(scored) != sorted(prefix):
            missing = sorted(set(prefix) - set(scored))
            extra = sorted(set(scored) - set(prefix))
            repeated = sorted({cid for cid in scored if scored.count(cid) > 1})
            problems.append(f"{where}/{name}: scored {len(scored)} of {len(prefix)} candidates"
                            + (f", missing {missing}" if missing else "")
                            + (f", scored outside the prefix {extra}" if extra else "")
                            + (f", scored twice {repeated}" if repeated else ""))
        else:
            expected = selector_choice(block["scores"], prefix)
            chosen = block.get("chosen") or {}
            if chosen.get("candidate_id") != expected["candidate_id"]:
                problems.append(f"{where}/{name}: chose {chosen.get('candidate_id')} but the locked "
                                f"tie-break picks {expected['candidate_id']}")
            if chosen.get("tie_break") != expected["tie_break"]:
                problems.append(f"{where}/{name}: tie_break {chosen.get('tie_break')!r} is not the "
                                f"recorded scores' {expected['tie_break']!r}")
        sources = block.get("information_sources")
        if not sources:
            problems.append(f"{where}/{name}: no information boundary declared")
        for source in sources or []:
            if source in FORBIDDEN_SOURCES:
                problems.append(f"{where}/{name}: declares {source}; the oracle and simulator "
                                f"state are never selector inputs")
            elif source not in INFORMATION_SOURCES:
                problems.append(f"{where}/{name}: unknown information source {source!r}")
        leaked = _leaked_keys(block)
        if leaked:
            problems.append(f"{where}/{name}: ground-truth fields inside the selector block: "
                            f"{', '.join(sorted(set(leaked))[:4])}")
        problems.extend(check_timing(scene, name, block))
    return problems


def check_run(artifact: dict) -> list[str]:
    """Everything a report is entitled to assume. An empty list means the run is usable."""
    problems = []
    metadata = artifact.get("metadata") or {}
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        problems.append(f"artifact_version {artifact.get('artifact_version')} is not {ARTIFACT_VERSION}")
    for key in REQUIRED_METADATA:
        if metadata.get(key) in (None, {}, []):
            problems.append(f"metadata is missing {key}")
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        problems.append(f"protocol version {metadata.get('protocol_version')} is not this code's "
                        f"{PROTOCOL_VERSION}")
    if metadata.get("program_schema_version") != SCHEMA_VERSION:
        problems.append("program schema version does not match this code")
    recorded = settings_hash({"success": metadata.get("success_definition"),
                              "oracle": metadata.get("oracle_definition"),
                              "primary_outcomes": list(metadata.get("primary_outcomes") or []),
                              "artifact_version": metadata.get("artifact_version")})
    if metadata.get("definition_hash") != recorded:
        problems.append("the success/oracle definitions were changed after the run was recorded")
    elif recorded != definition_hash():
        problems.append("the run's success/oracle definitions are not this code's; it measured "
                        "something else and must not be compared against a current run")
    if list(metadata.get("primary_outcomes") or []) != list(PRIMARY_OUTCOMES):
        problems.append(f"primary outcomes must stay the predeclared {list(PRIMARY_OUTCOMES)}")
    if metadata.get("git_dirty"):
        problems.append("the run was produced from a dirty worktree; its git SHA describes nothing")

    scenes = artifact.get("scenes") or []
    if not scenes:
        problems.append("the run has no scenes")
    if "excluded" not in artifact:
        problems.append("no exclusion list; every attempted scene must be accounted for")

    split = (metadata.get("dataset") or {}).get("split")
    seeds = set((metadata.get("dataset") or {}).get("scene_seeds") or [])
    if split in SPLITS:
        stray = sorted(seeds - set(SPLITS[split]))
        if stray:
            problems.append(f"seeds {stray[:4]} are outside the {split} split; train/test overlap")
        for other, declared in SPLITS.items():
            if other != split and seeds & set(declared):
                problems.append(f"the {split} run reuses {other} seeds: "
                                f"{sorted(seeds & set(declared))[:4]}")
    elif split is not None:
        problems.append(f"unknown split {split!r}")

    seen, budgets, pools = set(), set(), {}
    for scene in scenes:
        key = (scene.get("scenario_id"), scene.get("physics_seed"), scene.get("prefix"))
        if key in seen:
            problems.append(f"{scene.get('scenario_id')}: duplicated at prefix {scene.get('prefix')} "
                            f"and physics seed {scene.get('physics_seed')}")
        seen.add(key)
        if scene.get("scene_seed") not in seeds:
            problems.append(f"{scene.get('scenario_id')}: scene seed {scene.get('scene_seed')} is not "
                            f"in the declared dataset")
        budgets.add(json.dumps(scene.get("execution_budget"), sort_keys=True))
        pools.setdefault(scene.get("scenario_id"), set()).add(
            (scene.get("pool_id"), scene.get("observation_id")))
        problems.extend(check_scene(scene, metadata))
    if len(budgets) > 1:
        problems.append(f"{len(budgets)} different execution budgets in one run; selectors must "
                        f"share the budget to be comparable")
    for scenario, identities in pools.items():
        if len(identities) > 1:
            problems.append(f"{scenario}: scored against {len(identities)} different pool or "
                            f"observation identities")

    # Prefixes of one scenario must nest: prefix 16 is the first 16 of prefix 32, or the two
    # numbers are not measuring more of the same pool.
    by_scenario = {}
    for scene in scenes:
        by_scenario.setdefault((scene.get("scenario_id"), scene.get("physics_seed")), []).append(scene)
    for (scenario, _), group in by_scenario.items():
        group = sorted(group, key=lambda scene: scene.get("prefix") or 0)
        for smaller, larger in zip(group, group[1:]):
            head = (larger.get("pool_prefix") or [])[:len(smaller.get("pool_prefix") or [])]
            if head != (smaller.get("pool_prefix") or []):
                problems.append(f"{scenario}: prefix {larger.get('prefix')} is not nested inside "
                                f"prefix {smaller.get('prefix')}")
    return problems


# --------------------------------------------------------------------------- aggregation


def comparable_key(scene: dict, metadata: dict) -> tuple:
    """What must match before two records may be averaged together."""
    return (metadata.get("protocol_version"), metadata.get("program_schema_version"),
            metadata.get("definition_hash"), scene.get("scenario_id"), scene.get("observation_id"),
            scene.get("physics_seed"), scene.get("prefix"),
            json.dumps(scene.get("pool_prefix"), sort_keys=True),
            json.dumps(scene.get("execution_budget"), sort_keys=True))


class NotComparable(ValueError):
    """Aggregation across records that do not describe the same evaluation."""


def pairing_key(scene: dict) -> tuple:
    return (scene.get("scenario_id"), scene.get("physics_seed"), scene.get("prefix"))


def scene_metrics(scene: dict, name: str, threshold: float = 0.5) -> dict:
    """One selector on one scene. `selection_efficiency` is undefined when the pool has none."""
    block = scene["selectors"][name]
    chosen = block["chosen"]["candidate_id"]
    outcomes, prefix = scene["counterfactual"], scene["pool_prefix"]
    oracle = scene["oracle"]
    picked, best = outcomes[chosen], outcomes[oracle["candidate_id"]]
    has_success = scene["pool_has_success"]
    accepts = [(row, outcomes[row["candidate_id"]]) for row in block["scores"]
               if row.get("probability") is not None]
    return {
        "selected_success": bool(picked["success"]),
        "pool_has_success": bool(has_success),
        "selection_efficiency": (1.0 if picked["success"] else 0.0) if has_success else None,
        "oracle_gap": oracle["ranking"].index(chosen),
        "target_error_gap_mm": round(picked["final_target_error_mm"] - best["final_target_error_mm"], 3),
        "missed_available_success": bool(has_success and not picked["success"]),
        "false_accepts": sum(1 for row, outcome in accepts
                             if row["probability"] >= threshold and not outcome["success"]),
        "false_rejects": sum(1 for row, outcome in accepts
                             if row["probability"] < threshold and outcome["success"]),
        "brier": (round(sum((row["probability"] - float(outcome["success"])) ** 2
                            for row, outcome in accepts) / len(accepts), 4) if accepts else None),
        "mean_uncertainty": (round(sum(row.get("uncertainty") or 0.0 for row, _ in accepts)
                                   / len(accepts), 4) if accepts else None),
        "selector_seconds": block["timing"]["selector_seconds"],
        "cost_usd": block.get("cost_usd", 0.0),
        "candidates": len(prefix),
    }


def _mean(values: list) -> float | None:
    """Mean over the scenes where the metric is defined; None when it is defined nowhere."""
    values = [float(value) for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None


def _difference(mine, theirs) -> float | None:
    """A paired difference exists only where both arms defined the metric."""
    return None if mine is None or theirs is None else float(mine) - float(theirs)


def bootstrap_ci(values: list[float], resamples: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> list[float] | None:
    """Percentile CI over scenes, with a fixed seed so a report is reproducible."""
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples))
    low = means[int(alpha / 2 * resamples)]
    high = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return [round(low, 4), round(high, 4)]


def aggregate(artifact: dict, prefix: int | None = None) -> dict:
    """Per-selector means and paired differences, refusing anything that is not comparable."""
    problems = check_run(artifact)
    if problems:
        raise NotComparable(f"invalid benchmark artifact: {problems[0]}")
    metadata = artifact["metadata"]
    scenes = [scene for scene in artifact["scenes"]
              if prefix is None or scene["prefix"] == prefix]
    if not scenes:
        raise NotComparable(f"no scenes at prefix {prefix}")
    names = sorted(metadata["selectors"])

    cells = {}
    for scene in scenes:
        key = pairing_key(scene)
        if key in cells:
            raise NotComparable(f"{key} appears twice; paired metrics need one record per cell")
        cells[key] = scene
        for name in names:
            if name not in scene["selectors"]:
                raise NotComparable(f"{key}: {name} has no record; the comparison is not paired")
        if comparable_key(scene, metadata)[:3] != comparable_key(scenes[0], metadata)[:3]:
            raise NotComparable(f"{key}: protocol or definitions differ from the rest of the run")

    thresholds = {name: (metadata["selectors"][name] or {}).get("accept_threshold", 0.5)
                  for name in names}
    by_prefix, rows = {}, {}
    for scene in scenes:
        by_prefix.setdefault(scene["prefix"], []).append(scene)
        for name in names:
            rows[(pairing_key(scene), name)] = scene_metrics(scene, name, thresholds[name])

    reported = ("selected_success", "selection_efficiency", "oracle_gap", "target_error_gap_mm",
                "missed_available_success", "false_accepts", "false_rejects", "brier",
                "mean_uncertainty", "selector_seconds", "cost_usd")
    report = {"prefixes": {}, "paired": {}, "primary_outcomes": list(PRIMARY_OUTCOMES),
              "scenes": len(scenes), "excluded": artifact.get("excluded", [])}
    for size, group in sorted(by_prefix.items()):
        report["prefixes"][str(size)] = {
            "scenes": len(group),
            # Candidate-generation coverage, kept out of the selector block: no selector can
            # be blamed for a pool that never held a success.
            "pool_has_success": _mean([scene["pool_has_success"] for scene in group]),
            "selectors": {name: {metric: _mean([rows[(pairing_key(scene), name)][metric]
                                                for scene in group]) for metric in reported}
                          for name in names},
        }
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                pairs = {}
                for metric in PRIMARY_OUTCOMES:
                    differences = [_difference(rows[(pairing_key(scene), left)][metric],
                                               rows[(pairing_key(scene), right)][metric])
                                   for scene in group]
                    pairs[metric] = {"paired_scenes": sum(1 for value in differences if value is not None),
                                     "mean_difference": _mean(differences),
                                     "ci95": bootstrap_ci(differences)}
                report["paired"].setdefault(f"{left} - {right}", {})[str(size)] = pairs

    # Does the selector accept more failures as the pool grows? Predeclared, and reported
    # whichever way it comes out.
    report["false_accepts_by_prefix"] = {
        name: {str(size): _mean([rows[(pairing_key(scene), name)]["false_accepts"] for scene in group])
               for size, group in sorted(by_prefix.items())} for name in names}
    return report


# --------------------------------------------------------------------------- cli


def validate_paths(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            problems = check_run(json.loads(path.read_text()))
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            problems = [f"unreadable artifact: {type(error).__name__}: {error}"]
        total += len(problems)
        print(f"{'PASS' if not problems else 'FAIL'} {path}")
        for problem in problems:
            print(f"  ! {problem}")
    print(f"\n{len(paths)} runs checked, {total} problems")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", type=Path, nargs="+", help="benchmark artifacts to check")
    ap.add_argument("--contract", action="store_true", help="print the locked definitions and exit")
    args = ap.parse_args()
    if args.contract or not args.validate:
        print(json.dumps({"artifact_version": ARTIFACT_VERSION, "protocol_version": PROTOCOL_VERSION,
                          "program_schema_version": SCHEMA_VERSION,
                          "primary_outcomes": list(PRIMARY_OUTCOMES),
                          "outcome_fields": list(OUTCOME_FIELDS),
                          "success_definition": success_definition(),
                          "oracle_definition": oracle_definition(),
                          "timing_boundary": list(TIMING_BOUNDARY),
                          "excluded_from_selector": list(EXCLUDED_FROM_SELECTOR),
                          "definition_hash": definition_hash()}, indent=2))
        raise SystemExit(0)
    raise SystemExit(1 if validate_paths(args.validate) else 0)


if __name__ == "__main__":
    main()
