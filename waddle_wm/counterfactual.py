"""Execute every candidate from one identical MuJoCo state, and write the outcome records.

A verifier can only be judged against the best program that was actually available to it.
Running only the candidate a selector picked cannot show whether the pool held something
better, and running candidates in sequence without a state restore mixes one candidate's
debris into the next. So this module runs the counterfactual: every candidate in a cached
pool from #17, each one started from the same restored bytes.

    for each scene:
        snapshot the complete integration state once
        shuffle the candidates
        for each candidate:  restore -> execute -> record outcome
        hand the outcomes to benchmark_record, which owns the answer key

This module owns *execution* and nothing else. The oracle ordering, the artifact schema, the
selector tie-break, and the validator all live in `waddle_wm.benchmark_record` (#24); the
records written here are exactly its `OUTCOME_FIELDS`, and the artifact is what its
`check_run` validates. There is one definition of "best" in this repo and it is not here.

The answer key is *hidden*: it exists only after executing everything, which no deployable
verifier can do. `selector_view` is the only thing a selector is allowed to see, and
`check_execution` fails the run if a simulator coordinate or an outcome field appears in it.

    uv run python -m waddle_wm.counterfactual --pools data/pools --kind diagnostic
    uv run python -m waddle_wm.counterfactual --split test --physics-seeds 3 --perturbation-mm 5
    uv run python -m waddle_wm.counterfactual --validate data/counterfactual/train-diagnostic.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import mujoco
import numpy as np

from waddle_wm import program as prog
from waddle_wm.benchmark_record import (ARTIFACT_VERSION, LEAKED_FIELDS, OUTCOME_FIELDS,
                                        SceneRun, SelectorRun, Timing, oracle_best,
                                        run_metadata, selector_choice)
from waddle_wm.benchmark_record import check_run as check_artifact
from waddle_wm.perception import QUERIES as DETECTOR_QUERIES
from waddle_wm.pools import (DEFAULT_ROOT as POOL_ROOT, SPLITS, Scene, git_dirty, git_sha,
                             parse_seeds, settings_hash)
from waddle_wm.program import RANGES, ProgramError
from waddle_wm.sim import relling_scene as scene

DEFAULT_OUT = Path("data/counterfactual")
DEFAULT_TIMEOUT = 120.0
DEFAULT_PERTURBATION_MM = 3.0
# The preflight's own tolerance: re-running a candidate after every other one has run must
# reproduce its placement to here. Tighter than the oracle's 0.5 mm bucket on purpose — the
# claim being tested is that restoration is exact, not that it is close enough to sort by.
ORDER_TOLERANCE_MM = 0.1

# Fields the selector view must never carry, on top of the ones #24 already forbids inside a
# selector block. Checked by key, not by text: Claude's `note` is free prose and may well
# contain the word "success", which is not a leak. A *field* named after an outcome is.
HIDDEN_FIELDS = tuple(sorted({*LEAKED_FIELDS, *OUTCOME_FIELDS, "block_spawn", "snapshot",
                              "snapshot_id", "executions", "scenes", "restore_ok"}))
# What `first` and `random` were allowed to see. Neither reads the observation; declaring it
# states the boundary they were held to, which is what #24's check is about.
REFERENCE_SOURCES = ("observation_text",)


class ExecutionTimeout(RuntimeError):
    """A candidate ran past its wall-clock budget. Raised at the next rendered frame."""


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
    """Restore, execute one candidate, and report what physics did. No judgement here.

    The first nine keys are #24's `OUTCOME_FIELDS` minus `execution_order`, which only the
    scenario knows. The rest are this module's own evidence that the execution was fair.
    """
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
        # A declining candidate, a timeout and a crash all leave the object somewhere; that
        # is the honest final placement. `declined`, `timed_out` and `error` tell them apart.
        scene_obj.env.track_task(block, destination)
        state = scene_obj.env.state()
        success, max_lift = False, float(state["max_block_z"])
        target_error = float(state["target_distance"])
        failure_mode = "timeout" if timed_out else ("error" if error else "declined")

    return {"success": success, "failure_mode": failure_mode,
            "max_lift_mm": round(max_lift * 1000.0, 3),
            "final_target_error_mm": round(target_error * 1000.0, 3),
            "failed_attempts": sum(not episode.success for episode in episodes),
            "timed_out": timed_out, "error": error,
            "execution_seconds": round(seconds, 3),
            "declined": program.aborts is not None, "attempts": len(episodes),
            "restore_ok": restore_ok, "snapshot_id": snapshot["digest"],
            "sim_seconds": round(float(scene_obj.env.data.time) - sim_started, 3),
            "frames": scene_obj.env.frame_count}


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
    for candidate in pool["candidates"]:
        if len(programs) >= probes:
            break
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
        for name in ("final_target_error_mm", "max_lift_mm"):
            if abs(a[name] - b[name]) > ORDER_TOLERANCE_MM:
                mismatches.append(f"{cid}: {name} moved {abs(a[name] - b[name]):.3f} mm with execution order")
    scene_obj.restore()
    restored = first == second == scene_obj.snapshot_id
    reproduced = observation_id == pool["scene"]["observation_id"]
    return {"snapshot_id": scene_obj.snapshot_id,
            "restores_to_same_bytes": restored, "observation_reproduced": reproduced,
            "order_probes": len(programs), "order_tolerance_mm": ORDER_TOLERANCE_MM,
            "order_mismatches": mismatches,
            "ok": restored and reproduced and bool(programs) and not mismatches}


# --------------------------------------------------------------------------- selectors


def reference_rankings(pool: dict, scenario_id: str) -> dict[str, list[str]]:
    """The two selectors that need no verifier, so the artifact is never vacuous.

    `first` takes the earliest sample — what an agent that asks Claude once already does.
    `random` is the coin flip any ranking has to beat. #18's verifiers arrive through
    `--selections` and go through exactly the same scoring code.
    """
    ids = [candidate["candidate_id"] for candidate in pool["candidates"]]
    shuffled = list(ids)
    random.Random(scenario_id).shuffle(shuffled)
    return {"first": ids, "random": shuffled}


def selector_block(name: str, ranking: list[str], prefix: list[str]) -> dict:
    """One ranking, scored into the shape #24 validates.

    A ranking is turned into scores so it goes through the *locked* argmax and pool-index
    tie-break rather than a second, private notion of "the selector's pick". Its latency is
    measured across #24's boundary like any other selector's: for these two it is the cost
    of an index lookup, which is the honest number.
    """
    observation_ready_at = time.time()
    positions = {cid: index for index, cid in enumerate(ranking)}
    scores = [{"candidate_id": cid, "score": float(len(ranking) - positions.get(cid, len(ranking))),
               "probability": None, "uncertainty": None, "rank": positions.get(cid)}
              for cid in prefix]
    chosen_at = time.time()
    timing = Timing(observation_ready_at, chosen_at, {"rank_lookup": chosen_at - observation_ready_at})
    return SelectorRun(name, settings_hash({"selector": name, "kind": "reference_ranking"}),
                       list(REFERENCE_SOURCES), scores, selector_choice(scores, prefix),
                       timing.as_json()).as_json()


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


def scenario_id_of(pool: dict) -> str:
    """One identity for 'this pool, on this observation'. Physics seed pairs *within* it."""
    payload = f"{pool['pool_id']}:{pool['scene']['observation_id']}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def execute_scenario(scene_obj: Scene, pool: dict, physics_seed: int, sigma_mm: float,
                     timeout: float) -> tuple[dict, dict]:
    """Execute every candidate once, in a shuffled order, from one restored state."""
    snapshot = perturbed_snapshot(scene_obj, physics_seed, sigma_mm)
    programs = [(candidate, prog.validate_program(candidate["program"]))
                for candidate in pool["candidates"]]
    # Candidate identity travels in the record, so the order is free to be arbitrary.
    order = list(range(len(programs)))
    random.Random(f"{scenario_id_of(pool)}:{physics_seed}:order").shuffle(order)

    outcomes = {}
    for position, index in enumerate(order):
        candidate, program = programs[index]
        outcome = run_candidate(scene_obj, program, snapshot, scene_obj.observation, timeout)
        outcomes[candidate["candidate_id"]] = {**outcome, "execution_order": position,
                                               "candidate_index": candidate["index"],
                                               "diagnostic": candidate.get("diagnostic")}
    # The complete set, kept whole. #24's scenes carry pool *prefixes*, so on a pool whose
    # size is not a prefix boundary the widest scene sees fewer candidates than were executed
    # — and "every candidate has exactly one record" is a claim about all of them.
    facts = {"snapshot_id": snapshot["digest"], "physics_seed": physics_seed,
             "perturbation_mm": 0.0 if physics_seed == 0 else sigma_mm,
             "candidates": len(outcomes),
             "successes": sum(o["success"] for o in outcomes.values()),
             "declined": sum(o["declined"] for o in outcomes.values()),
             "errors": sum(1 for o in outcomes.values() if o["error"]),
             "outcomes": outcomes}
    return outcomes, facts


def scenes_for(pool: dict, outcomes: dict, physics_seed: int, budget: dict,
               selections: dict | None) -> list[dict]:
    """One `SceneRun` per pool prefix: the cell every paired metric in #24 is paired on."""
    scenario_id = scenario_id_of(pool)
    rankings = reference_rankings(pool, scenario_id)
    rankings.update((selections or {}).get(pool["pool_id"], {}))
    generation = {c["candidate_id"]: float((c.get("generation") or {}).get("seconds") or 0.0)
                  for c in pool["candidates"]}

    scenes = []
    for size in sorted(int(key) for key in pool["prefixes"]):
        prefix = pool["prefixes"][str(size)]
        inside = {cid: outcomes[cid] for cid in prefix}
        scenes.append(SceneRun(
            scenario_id=scenario_id, split=pool["split"], scene_seed=pool["scene"]["seed"],
            physics_seed=physics_seed, pool_id=pool["pool_id"], pool_kind=pool["kind"],
            observation_id=pool["scene"]["observation_id"], prefix=size, pool_prefix=list(prefix),
            counterfactual=inside,
            selectors={name: selector_block(name, ranking, prefix)
                       for name, ranking in rankings.items()},
            execution_budget=budget,
            claude_generation_seconds=sum(generation[cid] for cid in prefix),
            mujoco_execution_seconds=sum(inside[cid]["execution_seconds"] for cid in prefix),
        ).as_json())
    return scenes


def run_pools(pools: list[dict], split: str, kind: str, physics_seeds: int, sigma_mm: float,
              timeout: float, probes: int, selections: dict | None = None) -> tuple[dict, dict]:
    """Execute every pool and assemble one artifact `benchmark_record.check_run` can validate."""
    budget = {"candidate_timeout_s": timeout, "max_attempts": RANGES["max_attempts"][1],
              "perturbation_mm": sigma_mm, "physics_seeds": physics_seeds}
    scenes, preflights, execution, views, excluded = [], {}, {}, {}, []

    for pool in pools:
        suite = pool["scene"].get("suite") or {}
        scene_obj = Scene(pool["scene"]["seed"], suite.get("scene_spec"))
        scenario_id = scenario_id_of(pool)
        try:
            spawn = pool["scene"].get("block_spawn")
            if spawn and {k: [round(v, 5) for v in point] for k, point in spawn.items()} != scene_obj.blocks:
                excluded.append({"scenario_id": scenario_id, "pool_id": pool["pool_id"],
                                 "reason": "the scene seed no longer reproduces the pool's block spawn"})
                continue
            checks = preflight(scene_obj, pool, probes, timeout)
            preflights[scenario_id] = checks
            if not checks["ok"]:
                excluded.append({"scenario_id": scenario_id, "pool_id": pool["pool_id"],
                                 "reason": f"preflight failed: {checks}"})
                continue
            for physics_seed in range(physics_seeds):
                outcomes, facts = execute_scenario(scene_obj, pool, physics_seed, sigma_mm, timeout)
                execution.setdefault(scenario_id, []).append(facts)
                scenes.extend(scenes_for(pool, outcomes, physics_seed, budget, selections))
        finally:
            scene_obj.close()
        views[pool["pool_id"]] = selector_view(pool)

    used = [pool for pool in pools
            if not any(item["pool_id"] == pool["pool_id"] for item in excluded)]
    names = sorted({name for scene in scenes for name in scene["selectors"]})
    metadata = run_metadata(
        split=split,
        scene_seeds=[pool["scene"]["seed"] for pool in used],
        physics_seeds=list(range(physics_seeds)),
        pools={pool["pool_id"]: {"kind": pool["kind"], "seed": pool["scene"]["seed"],
                                 "observation_id": pool["scene"]["observation_id"],
                                 "candidates": len(pool["candidates"]),
                                 "generator_hash": pool["protocol"].get("generator_hash")}
               for pool in used},
        generator=(used[0]["protocol"]["generator"] if used else {}),
        perception={"detector_queries": list(DETECTOR_QUERIES), "camera": "demo",
                    "landing_pad": "given in the task frame, not detected"},
        physics={"perturbation_mm": sigma_mm, "physics_seeds": physics_seeds,
                 "candidate_timeout_s": timeout, "paired_by": "scenario",
                 "snapshot": "mjSTATE_INTEGRATION + model.site_pos, restored before every candidate"},
        selectors={name: {"kind": "reference_ranking" if name in ("first", "random") else "supplied",
                          "information_sources": list(REFERENCE_SOURCES)} for name in names},
        git_sha=git_sha(), git_dirty=git_dirty(),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    artifact = {"artifact_version": ARTIFACT_VERSION, "metadata": metadata, "scenes": scenes,
                "excluded": excluded, "kind": kind,
                "preflight": preflights, "execution": execution}
    return artifact, views


# --------------------------------------------------------------------------- integrity


def check_execution(artifact: dict, views: dict | None = None) -> list[str]:
    """The execution-side claims, on top of everything `benchmark_record.check_run` checks.

    #24 owns the artifact schema, the oracle and the selector tie-break. What it cannot see
    is whether the executions behind those records were fair: one restored snapshot, one
    record per candidate per physics seed, and an execution order that was a permutation
    rather than a subset.
    """
    problems = list(check_artifact(artifact))

    for scenario_id, checks in (artifact.get("preflight") or {}).items():
        if not checks.get("ok"):
            problems.append(f"{scenario_id}: preflight did not pass; outcomes are not comparable")
        if int(checks.get("order_probes") or 0) < 1:
            problems.append(f"{scenario_id}: preflight did not execute an order probe")

    # The fairness claims are about the complete execution set, not about the prefix slices
    # #24 reports from: one restored snapshot, one record per candidate, and an execution
    # order that was a permutation of the pool rather than a subset of it.
    sizes = {}
    for scenario_id, cells in (artifact.get("execution") or {}).items():
        for facts in cells:
            outcomes = facts.get("outcomes") or {}
            where = f"{scenario_id} (physics seed {facts.get('physics_seed')})"
            sizes.setdefault(scenario_id, set()).add(len(outcomes))
            if not outcomes:
                problems.append(f"{where}: no executions recorded")
                continue
            for key in OUTCOME_FIELDS:
                missing = [cid for cid, outcome in outcomes.items() if key not in outcome]
                if missing:
                    problems.append(f"{where}: {len(missing)} outcome records are missing {key}")
            orders = sorted(outcome.get("execution_order") for outcome in outcomes.values())
            if orders != list(range(len(outcomes))):
                problems.append(f"{where}: execution order is not a permutation of the pool")
            failed = [cid for cid, outcome in outcomes.items() if not outcome.get("restore_ok")]
            if failed:
                problems.append(f"{where}: restore mismatch before {sorted(failed)[:4]}")
            if len({outcome.get("snapshot_id") for outcome in outcomes.values()}) > 1:
                problems.append(f"{where}: candidates were not all started from one snapshot")

    for scenario_id, counts in sizes.items():
        if len(counts) > 1:
            problems.append(f"{scenario_id}: unequal pools across physics seeds {sorted(counts)}")

    # Every scene must be a slice of the executions it claims to summarise, and the oracle is
    # defined to take a success when one exists — if these disagree the ordering has been
    # edited into something that is not an answer key.
    for scene_record in artifact.get("scenes") or []:
        cells = (artifact.get("execution") or {}).get(scene_record["scenario_id"]) or []
        executed = next((facts.get("outcomes") or {} for facts in cells
                         if facts.get("physics_seed") == scene_record["physics_seed"]), None)
        where = f"{scene_record['scenario_id']} prefix {scene_record['prefix']}"
        if executed is None:
            problems.append(f"{where}: no execution set for physics seed "
                            f"{scene_record['physics_seed']}")
            continue
        unexecuted = [cid for cid in scene_record["pool_prefix"] if cid not in executed]
        if unexecuted:
            problems.append(f"{where}: scored candidates that were never executed {unexecuted[:4]}")
        elif any(executed[cid] != scene_record["counterfactual"].get(cid)
                 for cid in scene_record["pool_prefix"]):
            problems.append(f"{where}: the scene's outcomes differ from the recorded executions")
        # #24 has already reported an incomplete outcome record; asking the oracle to order a
        # hole would raise rather than add anything.
        if all(key in (scene_record["counterfactual"].get(cid) or {})
               for cid in scene_record["pool_prefix"] for key in OUTCOME_FIELDS):
            oracle = oracle_best(scene_record["pool_prefix"], scene_record["counterfactual"])
            if oracle["pool_has_success"] != any(scene_record["counterfactual"][cid]["success"]
                                                 for cid in scene_record["pool_prefix"]):
                problems.append(f"{where}: the oracle missed an available success")

    for pool_id, view in (views or {}).items():
        for name in sorted(set(_keys_in(view)) & set(HIDDEN_FIELDS)):
            problems.append(f"{pool_id}: the selector view exposes {name}")
        if not view.get("candidates"):
            problems.append(f"{pool_id}: the selector view has no candidates")
    return problems


def _keys_in(node) -> list[str]:
    if isinstance(node, dict):
        return [key for name, value in node.items() for key in (name, *_keys_in(value))]
    if isinstance(node, list):
        return [key for item in node for key in _keys_in(item)]
    return []


def pool_coverage(artifact: dict, pools: list[dict]) -> dict[str, bool]:
    """Whether the complete unperturbed pool, not merely its widest prefix, held a success."""
    coverage = {}
    for pool in pools:
        cells = (artifact.get("execution") or {}).get(scenario_id_of(pool), [])
        unperturbed = next((cell for cell in cells if cell.get("physics_seed") == 0), None)
        if unperturbed is not None:
            coverage[pool["pool_id"]] = any(outcome.get("success")
                                             for outcome in unperturbed["outcomes"].values())
    return coverage


# --------------------------------------------------------------------------- cli


def find_pools(root: Path, kind: str, split: str | None,
               seeds: list[int] | None) -> list[tuple[Path, dict]]:
    found = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "index.json":
            continue
        pool = json.loads(path.read_text())
        if pool["kind"] != kind or (split and pool["split"] != split):
            continue
        if seeds is not None and pool["scene"]["seed"] not in seeds:
            continue
        found.append((path, pool))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools", type=Path, default=POOL_ROOT, help="cached pools from #17")
    ap.add_argument("--split", choices=tuple(SPLITS), default="test")
    ap.add_argument("--kind", choices=("natural", "diagnostic", "stress"), default="natural",
                    help="one artifact per kind: the two have different generators")
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
    ap.add_argument("--allow-dirty", action="store_true",
                    help="execute from tracked edits; the artifact will fail validation, by design")
    ap.add_argument("--validate", type=Path, nargs="+", help="check artifacts and exit")
    args = ap.parse_args()

    if args.validate:
        total = 0
        for path in args.validate:
            artifact = json.loads(path.read_text())
            views_path = path.with_name(f"{path.stem}-views.json")
            views = json.loads(views_path.read_text()) if views_path.exists() else {}
            problems = check_execution(artifact, views)
            total += len(problems)
            print(f"{'PASS' if not problems else 'FAIL'} {path}")
            for problem in problems:
                print(f"  ! {problem}")
        print(f"\n{len(args.validate)} artifacts checked, {total} problems")
        raise SystemExit(1 if total else 0)

    if git_dirty() and not args.allow_dirty:
        raise SystemExit("refusing to execute from a dirty tracked worktree; commit the locked "
                         "code or pass --allow-dirty for an explicitly exploratory run")
    if args.preflight_probes < 1:
        raise SystemExit("--preflight-probes must be at least 1")

    seeds = parse_seeds(args.seeds) if args.seeds else None
    selections = json.loads(args.selections.read_text()) if args.selections else None
    found = find_pools(args.pools, args.kind, args.split, seeds)
    if not found:
        raise SystemExit(f"no {args.kind} pools under {args.pools} for split {args.split}")
    pools = [pool for _, pool in found]
    print(f"{len(pools)} {args.kind} pool(s), {sum(len(p['candidates']) for p in pools)} candidates, "
          f"{args.physics_seeds} physics seed(s)")

    artifact, views = run_pools(pools, args.split, args.kind, args.physics_seeds,
                                args.perturbation_mm, args.timeout, args.preflight_probes,
                                selections)
    problems = check_execution(artifact, views)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.split}-{args.kind}.json"
    path.write_text(json.dumps(artifact, indent=2, default=float) + "\n")
    path.with_name(f"{path.stem}-views.json").write_text(json.dumps(views, indent=2, default=float) + "\n")

    # Generation coverage belongs to the complete pool, so it goes back where #17 reserved
    # room for it. Physics seed 0 is canonical; perturbations are a robustness slice.
    if not problems:
        coverage = pool_coverage(artifact, pools)
        for pool_path, pool in found:
            if pool["pool_id"] in coverage:
                pool["pool_has_success"] = coverage[pool["pool_id"]]
                pool_path.write_text(json.dumps(pool, indent=2, default=float))

    for scenario_id, cells in artifact["execution"].items():
        head = cells[0]
        widest = max((s for s in artifact["scenes"]
                      if s["scenario_id"] == scenario_id and s["physics_seed"] == 0),
                     key=lambda s: s["prefix"])
        print(f"  {scenario_id}  snapshot {head['snapshot_id']}  "
              f"{head['successes']}/{head['candidates']} candidates succeed  "
              f"at prefix {widest['prefix']}: pool_has_success={widest['pool_has_success']}, "
              f"oracle={widest['oracle']['candidate_id']} "
              f"(decided by {widest['oracle']['decided_by']})")
    print(f"\nwrote {path} ({len(artifact['scenes'])} scenes, "
          f"{len(artifact['excluded'])} excluded)")
    for problem in problems:
        print(f"  ! {problem}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
