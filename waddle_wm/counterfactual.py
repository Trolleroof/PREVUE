"""Execute every candidate from one identical MuJoCo state and compute the hidden oracle.

A verifier can only be judged against the best program that was actually available to it.
Running only the candidate a selector picked shows whether *that* program worked; it cannot
show whether the pool held something better, and it cannot separate "the selector chose
badly" from "nothing in the pool would have worked". So this module runs the counterfactual:
every candidate in a cached pool from #17, each one started from the same restored bytes.

    for each scene:
        snapshot the complete integration state once
        shuffle the candidates
        for each candidate:  restore -> execute -> record outcome
        order the outcomes by the locked oracle key -> the answer key

The answer key is *hidden*. It is obtained only by executing everything, which no deployable
verifier can do, so it is never an input to Claude, to the estimated-state heuristic, or to
the visual verifier — `selector_view` is the only thing a selector is allowed to see, and
`check_run` fails if an outcome or a simulator coordinate ever appears inside it.

    uv run python -m waddle_wm.counterfactual --pools data/pools --kind diagnostic
    uv run python -m waddle_wm.counterfactual --split test --physics-seeds 3 --perturbation-mm 3
    uv run python -m waddle_wm.counterfactual --validate data/counterfactual
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from waddle_wm import program as prog
from waddle_wm.pools import DEFAULT_ROOT as POOL_ROOT, SPLITS, Scene, git_dirty, git_sha, parse_seeds
from waddle_wm.program import ProgramError
from waddle_wm.sim import relling_scene as scene

PROTOCOL_VERSION = 1
DEFAULT_OUT = Path("data/counterfactual")
DEFAULT_TIMEOUT = 120.0
DEFAULT_PERTURBATION_MM = 3.0

# The oracle ordering, locked here before any pool is executed. Nothing about a selector,
# a verifier, or a measured result may change it: it is the definition of "best available",
# not a knob. The last key is the pool index, so the ordering is total and reproducible
# rather than dependent on which candidate happened to be executed first.
ERROR_GRANULARITY_M = 1e-4        # 0.1 mm; two placements this close are the same placement
TIME_GRANULARITY_S = 0.1          # wall clock is noisy, so it decides only coarse differences
UNMEASURED_ERROR_M = 9.999        # an errored or timed-out run sorts behind every measured one
ORACLE_ORDERING = (
    {"key": "success", "direction": "descending",
     "note": "a candidate that did the task outranks every candidate that did not"},
    {"key": "failed_attempts", "direction": "ascending",
     "note": "a bounded retry that was needed is worse than one that was not"},
    {"key": "target_error_m", "direction": "ascending",
     "note": f"rounded to {ERROR_GRANULARITY_M} m"},
    {"key": "execution_seconds", "direction": "ascending",
     "note": f"rounded to {TIME_GRANULARITY_S} s"},
    {"key": "candidate_index", "direction": "ascending",
     "note": "pool order, so the ordering is total and independent of execution order"},
)

# Everything a selector is forbidden to see. `check_run` looks for these by name in the
# selector view, so a future field that leaks the answer key fails the benchmark's integrity
# check rather than quietly improving somebody's numbers.
HIDDEN_FIELDS = ("hidden_truth", "block_spawn", "snapshot", "snapshot_id", "executions",
                 "oracle", "pool_has_success", "success", "failure_mode", "target_error_m",
                 "max_lift_m", "failed_attempts", "selectors")


class ExecutionTimeout(RuntimeError):
    """A candidate ran past its wall-clock budget. Raised at the next rendered frame."""


@dataclass
class Execution:
    """One candidate, executed once, from one restored state. The oracle reads only these."""

    scenario_id: str
    pool_id: str
    snapshot_id: str
    physics_seed: int
    candidate_id: str
    candidate_index: int
    execution_order: int          # where in the shuffled sequence this candidate actually ran
    restore_ok: bool
    declined: bool
    success: bool
    failure_mode: str | None
    attempts: int
    failed_attempts: int
    max_lift_m: float | None
    target_error_m: float | None
    execution_seconds: float
    sim_seconds: float
    frames: int
    timed_out: bool
    error: str | None
    diagnostic: str | None = None
    strategy: str = ""


# --------------------------------------------------------------------------- execution


def _deadline_hook(deadline: float, budget: float):
    """A frame hook that turns a runaway execution into a recorded timeout, not a hung run."""
    def hook():
        if time.time() > deadline:
            raise ExecutionTimeout(f"candidate exceeded its {budget:g}s budget")
    return hook


def perturbed_snapshot(scene_obj: Scene, physics_seed: int, sigma_mm: float) -> dict:
    """The scenario's snapshot under one paired physics perturbation.

    Seed 0 is the snapshot itself. Higher seeds jitter every block by a seeded normal draw,
    and because the draw depends only on (scene seed, physics seed) the *same* perturbed
    scene is presented to every candidate — the pairing the comparison needs. The candidates
    were written against the unperturbed observation and are not told, which is the point:
    a program that redetects mid-flight can recover and one that binds once cannot.
    """
    if physics_seed == 0:
        return scene_obj.snapshot
    scene_obj.env.restore(scene_obj.snapshot)
    rng = np.random.default_rng([scene_obj.seed, physics_seed])
    for name in scene.BLOCK_NAMES:
        scene_obj.env.data.joint(f"{name}_free").qpos[:2] += rng.normal(0.0, sigma_mm / 1000.0, 2)
    mujoco.mj_forward(scene_obj.env.model, scene_obj.env.data)
    return scene_obj.env.snapshot()


def run_candidate(scene_obj: Scene, program: prog.Program, snapshot: dict,
                  observation, timeout: float) -> dict:
    """Restore, execute one candidate, and report what physics did. No judgement here."""
    restore_ok = scene_obj.restore(snapshot)
    block = program.object.replace(" ", "_")
    destination = program.destination.replace(" ", "_")
    started, sim_started = time.time(), float(scene_obj.env.data.time)
    scene_obj.env.on_frame = _deadline_hook(started + timeout, timeout)
    episodes, timed_out, error = [], False, None
    try:
        episodes = scene_obj.execute(program, observation)
    except ExecutionTimeout as failure:
        timed_out, error = True, str(failure)
    except (RuntimeError, ValueError, KeyError, ProgramError) as failure:
        error = f"{type(failure).__name__}: {failure}"
    finally:
        scene_obj.env.on_frame = None
    seconds = time.time() - started

    if episodes:
        last = episodes[-1]
        success = bool(last.success)
        failure_mode = last.failure_mode
        max_lift = max(float(episode.state_after["max_block_z"]) for episode in episodes)
        target_error = float(last.state_after["target_distance"])
    else:
        # A declining candidate and a crashed one both moved nothing, so the honest final
        # error is the untouched scene's. They are still told apart by `declined`.
        scene_obj.env.track_task(block, destination)
        state = scene_obj.env.state()
        success, max_lift = False, float(state["max_block_z"])
        target_error = None if error else float(state["target_distance"])
        failure_mode = "timeout" if timed_out else ("error" if error else "declined")

    return {"restore_ok": restore_ok, "declined": program.aborts is not None,
            "success": success, "failure_mode": failure_mode,
            "attempts": len(episodes),
            "failed_attempts": sum(not episode.success for episode in episodes),
            "max_lift_m": None if max_lift is None else round(max_lift, 6),
            "target_error_m": None if target_error is None else round(target_error, 6),
            "execution_seconds": round(seconds, 3),
            "sim_seconds": round(float(scene_obj.env.data.time) - sim_started, 3),
            "frames": scene_obj.env.frame_count, "timed_out": timed_out, "error": error}


def preflight(scene_obj: Scene, pool: dict, probes: int, timeout: float) -> dict:
    """Prove restoration works before spending a run on outcomes that assume it does.

    Three things, in order: the snapshot restores to the same bytes twice; the restored
    scene renders the same observation the pool was generated against; and executing the
    same few candidates in the opposite order gives the same outcomes. The last one is what
    makes a shuffled execution order safe — without a working restore, candidate k's result
    would carry candidate k-1's debris.
    """
    scene_obj.restore()
    first = scene_obj.env.state_digest()
    scene_obj.restore()
    second = scene_obj.env.state_digest()
    observation_id = scene_obj.observe().observation_id

    programs = []
    for candidate in pool["candidates"][:probes]:
        program = prog.validate_program(candidate["program"])
        if program.aborts is None:          # a declining candidate executes nothing to compare
            programs.append((candidate["candidate_id"], program))

    forward = {cid: run_candidate(scene_obj, program, scene_obj.snapshot, scene_obj.observation, timeout)
               for cid, program in programs}
    backward = {cid: run_candidate(scene_obj, program, scene_obj.snapshot, scene_obj.observation, timeout)
                for cid, program in reversed(programs)}

    mismatches = []
    for cid in forward:
        a, b = forward[cid], backward[cid]
        if a["success"] != b["success"] or a["failure_mode"] != b["failure_mode"]:
            mismatches.append(f"{cid}: {a['failure_mode']} forwards, {b['failure_mode']} backwards")
        for field_name, tolerance in (("target_error_m", ERROR_GRANULARITY_M), ("max_lift_m", ERROR_GRANULARITY_M)):
            x, y = a[field_name], b[field_name]
            if x is None or y is None:
                continue
            if abs(x - y) > tolerance:
                mismatches.append(f"{cid}: {field_name} moved {abs(x - y):.6f} with execution order")
    scene_obj.restore()
    return {"snapshot_id": scene_obj.snapshot_id,
            "restores_to_same_bytes": first == second == scene_obj.snapshot_id,
            "observation_reproduced": observation_id == pool["scene"]["observation_id"],
            "order_probes": len(programs), "order_tolerance_m": ERROR_GRANULARITY_M,
            "order_mismatches": mismatches,
            "ok": first == second == scene_obj.snapshot_id
                  and observation_id == pool["scene"]["observation_id"] and not mismatches}


# --------------------------------------------------------------------------- the oracle


def order_key(record: dict) -> tuple:
    """The locked oracle key. See ORACLE_ORDERING; lower is better on every component."""
    error = record["target_error_m"]
    error = UNMEASURED_ERROR_M if error is None else error
    return (not record["success"], record["failed_attempts"],
            round(error / ERROR_GRANULARITY_M),
            round(record["execution_seconds"] / TIME_GRANULARITY_S),
            record["candidate_index"])


def oracle_of(records: list[dict]) -> dict:
    """The best available candidate among `records`, and how firmly it won.

    `tied_with` is the number of other candidates the ordering could not separate before the
    pool-index tie-break. A selector that picks one of those has not chosen worse, which is
    why agreement on candidate id is reported but never used as the score.
    """
    if not records:
        return {"candidate_id": None, "candidate_index": None, "success": False,
                "target_error_m": None, "tied_with": 0, "ranking": []}
    ranked = sorted(records, key=order_key)
    best = ranked[0]
    tied = sum(1 for record in ranked[1:] if order_key(record)[:-1] == order_key(best)[:-1])
    return {"candidate_id": best["candidate_id"], "candidate_index": best["candidate_index"],
            "success": best["success"], "failure_mode": best["failure_mode"],
            "failed_attempts": best["failed_attempts"], "target_error_m": best["target_error_m"],
            "execution_seconds": best["execution_seconds"], "tied_with": tied,
            "ranking": [record["candidate_id"] for record in ranked]}


# --------------------------------------------------------------------------- selectors


def reference_rankings(pool: dict, scenario_id: str) -> dict[str, list[str]]:
    """The two selectors that need no verifier, so the report is never vacuous.

    `first` takes the earliest sample — what an agent that asks Claude once already does.
    `random` is the coin flip any ranking has to beat. #18's verifiers arrive through
    `--selections` and are scored by exactly the same code.
    """
    ids = [candidate["candidate_id"] for candidate in pool["candidates"]]
    shuffled = list(ids)
    random.Random(scenario_id).shuffle(shuffled)
    return {"first": ids, "random": shuffled}


def score_selector(ranking: list[str], records: list[dict], oracle: dict) -> dict:
    """Where one selector's pick lands against the answer key it never saw."""
    available = {record["candidate_id"]: record for record in records}
    chosen = next((cid for cid in ranking if cid in available), None)
    if chosen is None:
        return {"candidate_id": None, "success": False, "ranked_nothing_available": True}
    record = available[chosen]
    gap = None
    if record["target_error_m"] is not None and oracle["target_error_m"] is not None:
        gap = round(record["target_error_m"] - oracle["target_error_m"], 6)
    return {"candidate_id": chosen, "candidate_index": record["candidate_index"],
            "success": record["success"], "failure_mode": record["failure_mode"],
            "target_error_m": record["target_error_m"], "target_error_gap_m": gap,
            "oracle_rank": oracle["ranking"].index(chosen),
            "agrees_with_oracle": chosen == oracle["candidate_id"],
            "within_oracle_tie": order_key(record)[:-1] == order_key(available[oracle["candidate_id"]])[:-1],
            "missed_available_success": bool(oracle["success"] and not record["success"]),
            "ranked_nothing_available": False}


# --------------------------------------------------------------------------- one scenario


def selector_view(pool: dict) -> dict:
    """Everything a selector may see, and nothing else. This is #18's input.

    Deliberately reconstructed field by field rather than copied-and-deleted: a new key in
    the pool artifact has to be added here on purpose before any verifier can read it.
    """
    return {
        "pool_id": pool["pool_id"], "kind": pool["kind"], "split": pool["split"],
        "task": pool["task"],
        "scene": {"seed": pool["scene"]["seed"], "observation_id": pool["scene"]["observation_id"],
                  "observation": pool["scene"]["observation"],
                  "detections": pool["scene"]["detections"],
                  "landing_pad": pool["scene"].get("landing_pad")},
        "prefixes": pool["prefixes"],
        "candidates": [{"candidate_id": c["candidate_id"], "index": c["index"],
                        "program": c["program"], "grounded_trace": c["grounded_trace"],
                        "dedup_key": c["dedup_key"], "duplicate_of": c["duplicate_of"],
                        "retry": c["retry"], "redetect_ops": c["redetect_ops"],
                        "aborts": c.get("aborts"), "strategy": c.get("strategy", ""),
                        "note": c.get("note", "")}
                       for c in pool["candidates"]],
    }


def scenario_id_of(pool: dict, physics_seed: int) -> str:
    """One identity for 'this pool, from this state, under this perturbation'."""
    payload = f"{pool['pool_id']}:{pool['scene']['observation_id']}:{physics_seed}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def run_scenario(scene_obj: Scene, pool: dict, physics_seed: int, sigma_mm: float,
                 timeout: float, selections: dict | None = None) -> dict:
    """Execute every candidate once, in a shuffled order, from one restored state."""
    scenario_id = scenario_id_of(pool, physics_seed)
    snapshot = perturbed_snapshot(scene_obj, physics_seed, sigma_mm)
    programs = [(candidate, prog.validate_program(candidate["program"]))
                for candidate in pool["candidates"]]
    # Candidate identity travels with the candidate, so the order is free to be arbitrary.
    order = list(range(len(programs)))
    random.Random(f"{scenario_id}:order").shuffle(order)

    records: list[dict] = []
    for position, index in enumerate(order):
        candidate, program = programs[index]
        outcome = run_candidate(scene_obj, program, snapshot, scene_obj.observation, timeout)
        records.append(asdict(Execution(
            scenario_id=scenario_id, pool_id=pool["pool_id"], snapshot_id=snapshot["digest"],
            physics_seed=physics_seed, candidate_id=candidate["candidate_id"],
            candidate_index=candidate["index"], execution_order=position,
            diagnostic=candidate.get("diagnostic"), strategy=candidate.get("strategy", ""),
            **outcome)))
    records.sort(key=lambda record: record["candidate_index"])

    rankings = reference_rankings(pool, scenario_id)
    rankings.update((selections or {}).get(pool["pool_id"], {}))
    prefixes = {}
    for size in sorted(int(key) for key in pool["prefixes"]):
        inside = [record for record in records if record["candidate_index"] < size]
        oracle = oracle_of(inside)
        prefixes[str(size)] = {
            "candidates": len(inside),
            "pool_has_success": any(record["success"] for record in inside),
            "successes": sum(record["success"] for record in inside),
            "oracle": oracle,
            "selectors": {name: score_selector(ranking, inside, oracle)
                          for name, ranking in rankings.items()},
        }

    full = oracle_of(records)
    return {"scenario_id": scenario_id, "pool_id": pool["pool_id"], "kind": pool["kind"],
            "split": pool["split"], "scene_seed": pool["scene"]["seed"],
            "observation_id": pool["scene"]["observation_id"],
            "snapshot_id": snapshot["digest"], "physics_seed": physics_seed,
            "perturbation_mm": 0.0 if physics_seed == 0 else sigma_mm,
            "candidates": len(records),
            "pool_has_success": any(record["success"] for record in records),
            "successes": sum(record["success"] for record in records),
            "declined": sum(record["declined"] for record in records),
            "errors": sum(1 for record in records if record["error"]),
            "oracle": full, "oracle_ordering": list(ORACLE_ORDERING),
            "selector_rankings": rankings, "prefixes": prefixes, "executions": records}


def run_pool(pool: dict, physics_seeds: int, sigma_mm: float, timeout: float,
             probes: int, selections: dict | None = None) -> dict:
    """Every scenario for one cached pool: the preflight, then one pass per physics seed."""
    scene_obj = Scene(pool["scene"]["seed"])
    try:
        spawn = pool["scene"].get("block_spawn")
        if spawn and {k: [round(v, 5) for v in point] for k, point in spawn.items()} != scene_obj.blocks:
            raise ValueError(f"{pool['pool_id']}: the scene seed no longer reproduces the pool's block spawn")
        checks = preflight(scene_obj, pool, probes, timeout)
        started = time.time()
        scenarios = [run_scenario(scene_obj, pool, seed, sigma_mm, timeout, selections)
                     for seed in range(physics_seeds)]
    finally:
        scene_obj.close()
    return {"pool_id": pool["pool_id"], "kind": pool["kind"], "split": pool["split"],
            "scene_seed": pool["scene"]["seed"],
            "protocol": {"protocol_version": PROTOCOL_VERSION,
                         "program_schema_version": prog.SCHEMA_VERSION,
                         "git_sha": git_sha(), "git_dirty": git_dirty(),
                         "executed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "pool_generator_hash": pool["protocol"].get("generator_hash"),
                         "candidate_timeout_s": timeout, "physics_seeds": physics_seeds,
                         "perturbation_mm": sigma_mm,
                         "execution_seconds": round(time.time() - started, 1)},
            "oracle_ordering": list(ORACLE_ORDERING),
            "preflight": checks, "selector_view": selector_view(pool), "scenarios": scenarios}


# --------------------------------------------------------------------------- aggregate


def _rate(numerator: int, denominator: int) -> float | None:
    return None if not denominator else round(numerator / denominator, 4)


def aggregate(runs: list[dict]) -> dict:
    """The report the issue asks for, sliced by prefix and by selector.

    Generation coverage and selection quality are kept apart on purpose. `pass@N` is a fact
    about what Claude proposed; `oracle success@N` is the ceiling any selector could have
    reached; `selected success@N` is what one actually reached; and selection efficiency is
    the second over the first, so a pool with no success in it cannot make a selector look
    bad and a pool of nothing but successes cannot make one look good.
    """
    scenarios = [scenario for run in runs for scenario in run["scenarios"]]
    prefix_keys = sorted({key for scenario in scenarios for key in scenario["prefixes"]}, key=int)
    names = sorted({name for scenario in scenarios
                    for size in scenario["prefixes"].values() for name in size["selectors"]})

    generation, oracle_report = {}, {}
    for key in prefix_keys:
        rows = [scenario["prefixes"][key] for scenario in scenarios if key in scenario["prefixes"]]
        generation[key] = {"scenarios": len(rows),
                           "pass_at_n": _rate(sum(row["pool_has_success"] for row in rows), len(rows)),
                           "mean_successes": round(sum(row["successes"] for row in rows) / max(1, len(rows)), 2)}
        oracle_report[key] = {
            "success_at_n": _rate(sum(row["oracle"]["success"] for row in rows), len(rows)),
            "mean_target_error_m": _mean([row["oracle"]["target_error_m"] for row in rows]),
            "mean_tied_with": round(sum(row["oracle"]["tied_with"] for row in rows) / max(1, len(rows)), 2)}

    selectors = {}
    for name in names:
        by_prefix = {}
        for key in prefix_keys:
            rows = [(scenario["prefixes"][key], scenario["prefixes"][key]["selectors"][name])
                    for scenario in scenarios
                    if key in scenario["prefixes"] and name in scenario["prefixes"][key]["selectors"]]
            if not rows:
                continue
            winnable = [(row, pick) for row, pick in rows if row["oracle"]["success"]]
            by_prefix[key] = {
                "scenarios": len(rows),
                "success_at_n": _rate(sum(pick["success"] for _, pick in rows), len(rows)),
                # Conditional on the pool containing a success: the only slice where a
                # selector could have done anything, and therefore the only fair one.
                "winnable_scenarios": len(winnable),
                "selection_efficiency": _rate(sum(pick["success"] for _, pick in winnable), len(winnable)),
                "missed_available_successes": sum(pick["missed_available_success"] for _, pick in rows),
                "mean_target_error_gap_m": _mean([pick["target_error_gap_m"] for _, pick in rows]),
                "mean_oracle_rank": _mean([pick["oracle_rank"] for _, pick in rows]),
                "agrees_with_oracle": _rate(sum(pick["agrees_with_oracle"] for _, pick in rows), len(rows)),
                "within_oracle_tie": _rate(sum(pick["within_oracle_tie"] for _, pick in rows), len(rows)),
            }
        selectors[name] = by_prefix

    return {"pools": len({run["pool_id"] for run in runs}), "scenarios": len(scenarios),
            "physics_seeds": sorted({scenario["physics_seed"] for scenario in scenarios}),
            "executions": sum(len(scenario["executions"]) for scenario in scenarios),
            "splits": sorted({run["split"] for run in runs}),
            "kinds": sorted({run["kind"] for run in runs}),
            "oracle_ordering": list(ORACLE_ORDERING),
            "generation": generation, "oracle": oracle_report, "selectors": selectors,
            "integrity": [problem for run in runs for problem in check_run(run)]}


def _mean(values: list) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else round(sum(present) / len(present), 6)


# --------------------------------------------------------------------------- integrity


def check_run(run: dict) -> list[str]:
    """What a downstream ranking benchmark is entitled to assume. Empty list means usable."""
    problems = []
    view = run.get("selector_view", {})
    expected = [candidate["candidate_id"] for candidate in view.get("candidates", [])]
    if not expected:
        problems.append(f"{run.get('pool_id')}: the selector view has no candidates")
    if not run.get("preflight", {}).get("ok"):
        problems.append(f"{run.get('pool_id')}: preflight did not pass; outcomes are not comparable")

    sizes = set()
    for scenario in run.get("scenarios", []):
        where = f"{scenario['scenario_id']} (physics seed {scenario['physics_seed']})"
        records = scenario["executions"]
        ids = [record["candidate_id"] for record in records]
        sizes.add(len(records))
        if len(set(ids)) != len(ids):
            problems.append(f"{where}: a candidate was executed more than once")
        if sorted(ids) != sorted(expected):
            missing = sorted(set(expected) - set(ids))
            extra = sorted(set(ids) - set(expected))
            problems.append(f"{where}: candidates missing {missing or '-'}, unexpected {extra or '-'}")
        if sorted(record["execution_order"] for record in records) != list(range(len(records))):
            problems.append(f"{where}: execution order is not a permutation of the pool")
        if not all(record["restore_ok"] for record in records):
            failed = [r["candidate_id"] for r in records if not r["restore_ok"]]
            problems.append(f"{where}: restore mismatch before {failed}")
        if len({record["snapshot_id"] for record in records}) > 1:
            problems.append(f"{where}: candidates were not all started from one snapshot")
        if scenario["pool_has_success"] != any(record["success"] for record in records):
            problems.append(f"{where}: pool_has_success disagrees with the executions")
        # The oracle is defined to take a success when one exists; if these ever disagree the
        # ordering has been edited into something that is not an answer key.
        if scenario["oracle"]["success"] != scenario["pool_has_success"]:
            problems.append(f"{where}: the oracle missed an available success")
        if scenario["oracle_ordering"] != list(ORACLE_ORDERING):
            problems.append(f"{where}: the oracle ordering is not the locked one")
        for key, prefix in scenario["prefixes"].items():
            if prefix["oracle"]["success"] != prefix["pool_has_success"]:
                problems.append(f"{where}: prefix {key} oracle missed an available success")
            for name, pick in prefix["selectors"].items():
                if pick["candidate_id"] is not None and pick["candidate_index"] >= int(key):
                    problems.append(f"{where}: selector {name} chose outside prefix {key}")

    if len(sizes) > 1:
        problems.append(f"{run.get('pool_id')}: unequal pools across physics seeds {sorted(sizes)}")

    # Structural, not textual: Claude's `note` is free prose and may well contain the word
    # "success", which is not a leak. A *key* named after an outcome is.
    for name in sorted(set(_keys_in(view)) & set(HIDDEN_FIELDS)):
        problems.append(f"{run.get('pool_id')}: the selector view exposes {name}")
    return problems


def _keys_in(node) -> list[str]:
    if isinstance(node, dict):
        return [key for name, value in node.items() for key in (name, *_keys_in(value))]
    if isinstance(node, list):
        return [key for item in node for key in _keys_in(item)]
    return []


def validate_root(root: Path) -> int:
    """Check every counterfactual run under `root`; return the number of problems found."""
    total = 0
    files = [path for path in sorted(root.rglob("*.json")) if path.name != "summary.json"]
    for path in files:
        problems = check_run(json.loads(path.read_text()))
        total += len(problems)
        print(f"{'PASS' if not problems else 'FAIL'} {path}: {'ok' if not problems else '; '.join(problems[:4])}")
    print(f"\n{len(files)} runs checked, {total} problems")
    return total


# --------------------------------------------------------------------------- cli


def find_pools(root: Path, kinds: tuple[str, ...], split: str | None,
               seeds: list[int] | None) -> list[Path]:
    paths = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "index.json":
            continue
        pool = json.loads(path.read_text())
        if pool["kind"] not in kinds:
            continue
        if split and pool["split"] != split:
            continue
        if seeds is not None and pool["scene"]["seed"] not in seeds:
            continue
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools", type=Path, default=POOL_ROOT, help="cached pools from #17")
    ap.add_argument("--split", choices=tuple(SPLITS), default=None, help="only pools from this split")
    ap.add_argument("--kind", choices=("natural", "diagnostic", "both"), default="both")
    ap.add_argument("--seeds", help="explicit scene seeds, e.g. 0,1,2 or 100-107")
    ap.add_argument("--physics-seeds", type=int, default=1,
                    help="paired perturbations per pool; seed 0 is the snapshot itself")
    ap.add_argument("--perturbation-mm", type=float, default=DEFAULT_PERTURBATION_MM,
                    help="block jitter for physics seeds above 0, identical across candidates")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="wall clock per candidate")
    ap.add_argument("--preflight-probes", type=int, default=3,
                    help="candidates re-executed in reverse order to prove order independence")
    ap.add_argument("--selections", type=Path,
                    help='{"<pool_id>": {"<selector>": ["<candidate_id>", ...]}} from #18')
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--rerun", action="store_true", help="ignore cached runs and execute again")
    ap.add_argument("--validate", type=Path, help="check cached runs under this directory and exit")
    args = ap.parse_args()

    if args.validate is not None:
        raise SystemExit(1 if validate_root(args.validate) else 0)

    kinds = ("natural", "diagnostic") if args.kind == "both" else (args.kind,)
    seeds = parse_seeds(args.seeds) if args.seeds else None
    selections = json.loads(args.selections.read_text()) if args.selections else None
    paths = find_pools(args.pools, kinds, args.split, seeds)
    if not paths:
        raise SystemExit(f"no pools under {args.pools} match kind={args.kind} split={args.split}")

    runs = []
    for path in paths:
        pool = json.loads(path.read_text())
        out_path = args.out / path.relative_to(args.pools)
        if out_path.exists() and not args.rerun:
            cached = json.loads(out_path.read_text())
            if cached["protocol"].get("pool_generator_hash") == pool["protocol"].get("generator_hash"):
                print(f"cached  {out_path}")
                runs.append(cached)
                continue
            print(f"stale   {out_path}  (the pool was regenerated; executing again)")

        run = run_pool(pool, args.physics_seeds, args.perturbation_mm, args.timeout,
                       args.preflight_probes, selections)
        problems = check_run(run)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(run, indent=2, default=float))
        runs.append(run)

        # Generation coverage belongs to the pool, so it goes back where #17 reserved room
        # for it. Physics seed 0 is the canonical scene; the perturbations are a robustness
        # slice, not a restatement of what Claude proposed.
        pool["pool_has_success"] = run["scenarios"][0]["pool_has_success"]
        path.write_text(json.dumps(pool, indent=2, default=float))

        head = run["scenarios"][0]
        print(f"{'wrote  ' if not problems else 'PROBLEM'} {out_path}  "
              f"{head['successes']}/{head['candidates']} succeed, "
              f"pool_has_success={head['pool_has_success']}, "
              f"oracle={head['oracle']['candidate_id']} "
              f"(tied with {head['oracle']['tied_with']})")
        for problem in problems:
            print(f"  ! {problem}")

    summary = aggregate(runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print("\n" + json.dumps({key: summary[key] for key in
                             ("pools", "scenarios", "executions", "generation", "oracle", "selectors")},
                            indent=2, default=float))
    print(f"wrote {args.out / 'summary.json'}")
    for problem in summary["integrity"]:
        print(f"  ! {problem}")
    raise SystemExit(1 if summary["integrity"] else 0)


if __name__ == "__main__":
    main()
