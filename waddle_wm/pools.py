"""Cached candidate pools: independent Claude program samples, plus a scripted fault suite.

One command per split produces, for every scene seed, two artifacts:

* a **natural** pool — up to 64 independent `claude -p` samples of the program schema in
  `waddle_wm.program`, drawn under one identical model, prompt, and observation, kept in
  sample order so every downstream verifier ranks the same nested prefixes;
* a **diagnostic** pool — the named strategies and planted faults from
  `program.diagnostic_programs`, deterministic and Claude-free, reported separately because a
  planted bug says nothing about what Claude proposes.

    uv run python -m waddle_wm.pools --split test --pool-size 64
    uv run python -m waddle_wm.pools --seeds 0,1,2 --pool-size 8 --model claude-haiku-4-5-20251001
    uv run python -m waddle_wm.pools --validate data/pools

Pools are generated once and cached. Everything downstream (#23 execution, #18 ranking,
#26 search) reads the cache and never regenerates, so a verifier can never be compared
against a pool it helped shape. Claude sees the camera observation and nothing else: no
MuJoCo state, no outcome, no label.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from waddle_wm import program as prog
from waddle_wm.perception import QUERIES as DETECTOR_QUERIES, SceneCamera, landing_pad
from waddle_wm.planner import ClaudePlanner, MODEL, describe_observation
from waddle_wm.program import MAX_OPS, RANGES, SCHEMA_VERSION, ProgramError, SceneObservation
from waddle_wm.sim import relling_scene as scene
from waddle_wm.sim.env import TabletopEnv

PROTOCOL_VERSION = 1
DEFAULT_ROOT = Path("data/pools")
POOL_SIZE = 64
PREFIXES = (1, 4, 16, 32, 64)
# Disjoint scene seeds. Nothing tuned on `test` may have been fitted on the others.
SPLITS = {"train": range(0, 40), "calibration": range(40, 60), "test": range(100, 160)}

SYSTEM_PROMPT = f"""You write short robot policy programs for a UR5e arm with a Robotiq 2F-85 gripper on a
tabletop. You choose the strategy and the numbers; a deterministic compiler turns your program into
Cartesian waypoints, solves IK, and runs it. You never see the outcome.

Reply with ONE JSON object and nothing else. No prose, no markdown fence.

{{"schema_version": {SCHEMA_VERSION},
  "strategy": "<a few words naming your approach>",
  "task": {{"object": "<red block|blue block|yellow block>",
           "destination": "<green pad|red block|blue block|yellow block>"}},
  "ops": [ ... ],
  "note": "<one or two sentences on what you chose and why>"}}

The complete API — any other operation is rejected:

  {{"op": "detect", "query": "<red block|blue block|yellow block|green pad>", "as": "<symbol>"}}
      Look the thing up with the camera and bind its centre to a symbol. Every later
      operation refers to symbols, never to raw coordinates. Detecting again rebinds the
      symbol to wherever the thing is at that point in the program.
  {{"op": "move_above", "ref": "<symbol>", "offset_mm": [dx, dy], "height_mm": h,
   "direction": "<top|+x|-x|+y|-y>", "standoff_mm": s, "yaw_deg": y}}
      Move the pinch point to the symbol, offset laterally by offset_mm, and s mm further
      along `direction` ("top" means straight over it), at height h above the table.
  {{"op": "descend_to", "ref": "<symbol>", "offset_mm": [dx, dy], "height_mm": h, "yaw_deg": y}}
      Come down over the symbol to height h above the table.
  {{"op": "grasp"}}    close the gripper
  {{"op": "release"}}  open the gripper
  {{"op": "lift", "height_mm": h}}      rise straight up to h above the table
  {{"op": "retreat", "height_mm": h}}   rise straight up to h above the table, at the end
  {{"op": "on_failure", "policy": "<abort|redetect_regrasp>", "max_attempts": 0..2}}
      Optional, last operation only. What to do if the object is not held after the lift.
  {{"op": "abort", "reason": "<why>"}}
      Decline to act. Only detect() may come before it and nothing may come after. Use it
      when the task names something the observation does not contain, or when acting would
      be unsafe. Declining is a real answer, not a failure — but do not use it to avoid a
      task the scene supports.

`yaw_deg` is optional on move_above and descend_to: the heading of the gripper's jaws about
the vertical, in degrees, held until you change it. Leave it out and the wrist settles
wherever the solver puts it; set it to grasp a rotated object along its short side. The jaws
are symmetric, so -90..90 covers every distinct heading.

Bounds — a program outside them is rejected without ever being run:
- offset_mm and standoff_mm components: {RANGES['offset_mm'][0]:.0f}..{RANGES['offset_mm'][1]:.0f} mm.
- move_above / lift / retreat height_mm: {RANGES['hover_mm'][0]:.0f}..{RANGES['hover_mm'][1]:.0f} mm.
- descend_to height_mm: {RANGES['grasp_mm'][0]:.0f}..{RANGES['grasp_mm'][1]:.0f} mm before the grasp,
  {RANGES['release_mm'][0]:.0f}..{RANGES['release_mm'][1]:.0f} mm after it.
- At most {MAX_OPS} operations, and at most 8 of them may be motion or gripper operations.
- Retries are bounded: max_attempts is at most {RANGES['max_attempts'][1]}.
- You must grasp() exactly what the task names and release() before the program ends, unless
  you abort().

Geometry you can rely on: the blocks are cubes 36 mm on a side sitting on the table, so their
centres are 18 mm up and their tops are 36 mm up. To stack one block on another, release above
the destination block's top. The green pad is flat.

You are one of many independent proposals for the same scene. Commit to a single concrete
program — do not hedge, do not explain alternatives, do not ask questions."""


def sample_prompt(instruction: str, observation: str) -> str:
    return (f"Task instruction (from the operator):\n{instruction}\n\n"
            f"Camera observation of the scene right now:\n{observation}\n\n"
            "Write the program. Reply with the JSON object only.")


@dataclass
class Candidate:
    candidate_id: str
    index: int                    # position in the ranked pool; prefixes are `index < N`
    sample_index: int             # which generation attempt produced it
    program: dict
    grounded_trace: list[dict]
    dedup_key: str
    duplicate_of: str | None
    validation: dict
    retry: dict
    redetect_ops: list[int]
    aborts: str | None = None     # the reason, when the candidate declines to act
    diagnostic: str | None = None      # which named strategy or fault this is, in a diagnostic pool
    diagnostic_kind: str | None = None # "strategy" or "fault"
    strategy: str = ""
    note: str = ""
    generation: dict = field(default_factory=dict)
    raw: str = ""


@dataclass
class Rejected:
    sample_index: int
    stage: str                    # generation | schema | grounding | reachability
    error: str
    raw: str = ""
    generation: dict = field(default_factory=dict)


def settings_hash(settings: dict) -> str:
    return hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return True


class Scene:
    """One seeded tabletop, its rendered observation, and the reachability check for it."""

    def __init__(self, seed: int, scene_spec: dict | None = None):
        self.seed = seed
        params = scene_spec["scene_parameters"] if scene_spec else None
        self.env = TabletopEnv(
            seed=seed,
            block_sizes={"red_block": params["red_block_size"]} if params else None,
            block_masses={"red_block": params["red_block_mass_kg"]} if params else None,
        )
        self.blocks = {name: [round(float(v), 5) for v in point]
                       for name, point in self.env.sample_blocks().items()}
        self.env.reset(blocks=self.blocks)
        if scene_spec is not None:
            from waddle_wm.perception_scenes import apply_scene_spec
            apply_scene_spec(self.env, scene_spec)
            self.blocks = {name: [round(float(v), 5) for v in point]
                           for name, point in self.env.block_positions().items()}
        self.camera = SceneCamera(self.env.model, self.env.data)
        self.observation = self.observe()
        self.text = self.observation.text
        self.truth = {name: [round(float(v), 4) for v in point]
                      for name, point in self.env.block_positions().items()}
        # The exact bytes the observation was taken from. Every replay starts here, so a
        # candidate is never advantaged or handicapped by whatever ran before it.
        self.snapshot = self.env.snapshot()
        self.snapshot_id = self.snapshot["digest"]

    def observe(self) -> SceneObservation:
        """Read the current camera scene; used initially and at live redetect operations."""
        detections = self.camera.detect_all(DETECTOR_QUERIES)
        pad, gripper = landing_pad(self.env.model), self.env.home_waypoint()
        text = describe_observation(detections, pad, gripper)
        return SceneObservation.from_perception(detections, pad, gripper, self.seed, text)

    def restore(self, snapshot: dict | None = None) -> bool:
        """Put the tabletop back exactly where the observation was taken, for a replay.

        Returns whether the restored state fingerprint matches the snapshot's own, so a
        caller can record a failed restore instead of quietly comparing two different scenes.
        """
        snapshot = self.snapshot if snapshot is None else snapshot
        self.env.restore(snapshot)
        return self.env.state_digest() == snapshot["digest"]

    def reachable(self, trace) -> str | None:
        """Walk the trace through the same damped-least-squares IK the executor uses."""
        q = np.array([self.env.data.joint(joint).qpos[0] for joint in scene.ARM_JOINTS])
        for entry in trace:
            if "target" not in entry:
                continue
            try:
                q = self.env._ik(entry["target"], q, entry.get("yaw"))
            except RuntimeError as error:
                return f"{entry['phase']} waypoint is unreachable: {error}"
        return None

    def execute(self, program: prog.Program, observation: SceneObservation | None = None) -> list:
        """Replay one complete policy, including live redetection and bounded retries.

        `observation` is what the program's symbols bind to at the start. It defaults to a
        fresh look at the scene; the counterfactual benchmark passes the pool's own
        observation instead, so a perturbed replay cannot silently re-ground onto the
        perturbation the candidate was never told about.
        """
        block, destination = program.object.replace(" ", "_"), program.destination.replace(" ", "_")

        def run_attempt(segments):
            self.env.clear_recording()
            return self.env.run_trace_segments(segments, block=block, destination=destination,
                                               params={"program": program.strategy})

        return prog.execute(program, observation or self.observe(), self.observe, run_attempt)

    def close(self):
        self.camera.close()


def admit(scene_obj: Scene, program: prog.Program, sample_index: int, keys: dict,
          generation: dict, diagnostic: str | None = None) -> tuple[Candidate | None, Rejected | None]:
    """Ground, check reachability, and dedup one program against the pool built so far."""
    try:
        grounded = prog.ground(program, scene_obj.observation)
    except ProgramError as error:
        return None, Rejected(sample_index, "grounding", str(error), program.raw, generation)
    trace = grounded.step.summary()["trace"] if grounded.step else []
    unreachable = scene_obj.reachable(grounded.trace)
    if unreachable:
        return None, Rejected(sample_index, "reachability", unreachable, program.raw, generation)

    key = grounded.dedup_key()
    payload = json.dumps(program.as_json(), sort_keys=True)
    candidate_id = hashlib.sha1(f"{scene_obj.observation.observation_id}:{sample_index}:{payload}"
                                .encode()).hexdigest()[:12]
    return Candidate(candidate_id, -1, sample_index, program.as_json(), trace, key, keys.get(key),
                     {"ok": True, "stage": "accepted", "error": None},
                     program.retry, program.redetects, program.aborts, diagnostic, None,
                     program.strategy, program.note, generation, program.raw), None


def one_sample(index: int, instruction: str, observation_text: str, model: str, timeout: float) -> dict:
    """One independent `claude -p` call. Never re-asked: every candidate gets the same budget."""
    caller = ClaudePlanner(model=model, timeout=timeout, retries=0, system_prompt=SYSTEM_PROMPT)
    started = time.time()
    try:
        raw = caller.complete(sample_prompt(instruction, observation_text))
    except (RuntimeError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {"index": index, "error": f"{type(error).__name__}: {error}", "raw": "",
                "generation": {"model": model, "seconds": round(time.time() - started, 2)}}
    call = caller.calls[-1] if caller.calls else {}
    return {"index": index, "error": None, "raw": raw,
            "generation": {"model": model, "cost_usd": call.get("cost_usd"),
                           "duration_ms": call.get("duration_ms"), "session_id": call.get("session_id"),
                           "seconds": round(time.time() - started, 2)}}


def natural_pool(scene_obj: Scene, instruction: str, object_name: str, destination: str,
                 size: int, model: str, workers: int, oversample: float, timeout: float,
                 accepted: list[Candidate] | None = None,
                 rejected: list[Rejected] | None = None) -> tuple[list, list]:
    """Independent samples until `size` of them are valid, in sample order."""
    accepted, rejected = list(accepted or []), list(rejected or [])
    keys = {}
    for candidate in accepted:
        keys.setdefault(candidate.dedup_key, candidate.candidate_id)
    budget = max(size, int(round(size * oversample)))
    attempted = [item.sample_index for item in (*accepted, *rejected)]
    issued = max(attempted, default=-1) + 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(accepted) < size and issued < budget:
            # Never issue more paid calls than there are open candidate slots. Rejections
            # simply trigger another batch; every completed call is therefore recorded.
            batch = min(workers, budget - issued, size - len(accepted))
            futures = [pool.submit(one_sample, issued + offset, instruction, scene_obj.observation.text,
                                   model, timeout) for offset in range(batch)]
            issued += batch
            for item in (future.result() for future in futures):
                if item["error"]:
                    rejected.append(Rejected(item["index"], "generation", item["error"], "", item["generation"]))
                    continue
                try:
                    program = prog.parse(item["raw"])
                except (ProgramError, ValueError, TypeError, KeyError) as error:
                    # One malformed reply is a rejected sample, never a lost pool: the run
                    # is long and paid for, so anything unparseable is recorded and skipped.
                    rejected.append(Rejected(item["index"], "schema", f"{type(error).__name__}: {error}",
                                             item["raw"], item["generation"]))
                    continue
                if program.object != object_name or program.destination != destination:
                    rejected.append(Rejected(item["index"], "schema",
                                             f"program targets {program.object} -> {program.destination}, "
                                             f"the task is {object_name} -> {destination}",
                                             item["raw"], item["generation"]))
                    continue
                candidate, reject = admit(scene_obj, program, item["index"], keys, item["generation"])
                if reject is not None:
                    rejected.append(reject)
                    continue
                keys.setdefault(candidate.dedup_key, candidate.candidate_id)
                accepted.append(candidate)
    for position, candidate in enumerate(accepted):
        candidate.index = position
    return accepted, rejected


def diagnostic_pool(scene_obj: Scene, object_name: str, destination: str) -> tuple[list, list]:
    """The scripted suite: deterministic, no Claude, one entry per named strategy and fault."""
    shift = np.random.default_rng(scene_obj.seed).uniform(30.0, 60.0, 2) * np.array([1.0, -1.0])
    accepted, rejected, keys = [], [], {}
    for index, (name, kind, program) in enumerate(prog.diagnostic_programs(
            object_name, destination, stale_shift_mm=tuple(shift.round(1)))):
        candidate, reject = admit(scene_obj, program, index, keys,
                                  {"model": "scripted", "cost_usd": 0.0}, diagnostic=name)
        if reject is not None:
            reject.stage = f"{reject.stage} ({name})"
            rejected.append(reject)
            continue
        candidate.diagnostic_kind = kind
        keys.setdefault(candidate.dedup_key, candidate.candidate_id)
        accepted.append(candidate)
    for position, candidate in enumerate(accepted):
        candidate.index = position
    return accepted, rejected


@lru_cache(maxsize=1)
def claude_cli_version() -> str:
    try:
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=10)
        return (result.stdout or result.stderr).strip().splitlines()[0][:120] or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def generator_code_hash() -> str:
    functions = (prog.validate_program, prog.ground, prog.trace_segments, prog.execute,
                 prog.GroundedProgram.dedup_key, Scene.observe, Scene.reachable, admit, natural_pool)
    return hashlib.sha1("\n".join(inspect.getsource(function) for function in functions).encode()).hexdigest()[:12]


def generator_settings(kind: str, model: str, size: int, instruction: str,
                       workers: int, oversample: float, timeout: float) -> dict:
    """Everything that decides what a candidate looks like, so a cached pool can be trusted."""
    return {"kind": kind, "model": model if kind == "natural" else "scripted",
            "instruction": instruction,
            "system_prompt_sha1": hashlib.sha1(SYSTEM_PROMPT.encode()).hexdigest()[:12],
            "prompt_template_sha1": hashlib.sha1(sample_prompt("", "").encode()).hexdigest()[:12],
            "sampling": {"temperature": "claude CLI default", "max_output_tokens": "claude CLI default",
                         "stateless": True, "tools": False, "retry_bad_sample": False},
            "claude_cli_version": claude_cli_version() if kind == "natural" else "not-used",
            "parameter_ranges": RANGES, "generator_code_sha1": generator_code_hash(),
            "workers": workers, "oversample": oversample, "timeout": timeout,
            "attempt_budget": max(size, int(round(size * oversample))),
            "max_turns": 1, "pool_size": size, "program_schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION}


def generator_hash(settings: dict) -> str:
    """The settings hash the cache keys on. `pool_size` is excluded: a larger pool of the same
    kind extends a smaller one, but a changed prompt, model, or schema does not."""
    return settings_hash({key: value for key, value in settings.items()
                          if key not in ("pool_size", "attempt_budget")})


def build_pool(scene_obj: Scene, kind: str, instruction: str, object_name: str, destination: str,
               size: int, model: str, split: str, workers: int, oversample: float,
               timeout: float, previous: dict | None = None, scene_record: dict | None = None,
               manifest_lock: str | None = None) -> dict:
    settings = generator_settings(kind, model, size, instruction, workers, oversample, timeout)
    started = time.time()
    if kind == "natural":
        old_candidates = [Candidate(**candidate) for candidate in previous["candidates"]] if previous else None
        old_rejected = [Rejected(**item) for item in previous["rejected"]] if previous else None
        accepted, rejected = natural_pool(scene_obj, instruction, object_name, destination,
                                          size, model, workers, oversample, timeout,
                                          old_candidates, old_rejected)
    else:
        accepted, rejected = diagnostic_pool(scene_obj, object_name, destination)
    pool_id = (f"{scene_record['program_pool_id']}-{kind}" if scene_record else
               f"{kind}-{object_name.replace(' ', '_')}-to-{destination.replace(' ', '_')}"
               f"-seed{scene_obj.seed:04d}-{generator_hash(settings)}")

    unique = {}
    for candidate in accepted:
        unique.setdefault(candidate.dedup_key, candidate.candidate_id)
    pool = {
        "pool_id": pool_id, "kind": kind, "split": split,
        "protocol": {"protocol_version": PROTOCOL_VERSION, "program_schema_version": SCHEMA_VERSION,
                     "git_sha": git_sha(), "git_dirty": git_dirty(),
                     "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "generator": settings, "generator_hash": generator_hash(settings),
                     "generation_seconds": round((previous or {}).get("protocol", {}).get(
                         "generation_seconds", 0.0) + time.time() - started, 1)},
        "task": {"instruction": instruction, "object": object_name, "destination": destination},
        "scene": {"seed": scene_obj.seed, "observation_id": scene_obj.observation.observation_id,
                  "observation": scene_obj.observation.text,
                  "detections": scene_obj.observation.detections,
                  "landing_pad": {"centre": scene_obj.observation.points["green pad"][:2],
                                  "radius": scene_obj.observation.pad_radius},
                  "hidden_truth": scene_obj.truth, "block_spawn": scene_obj.blocks},
        "candidates": [asdict(candidate) for candidate in accepted],
        "rejected": [asdict(item) for item in rejected],
        "prefixes": {str(n): [c.candidate_id for c in accepted[:n]] for n in PREFIXES if n <= len(accepted)},
        # Filled in by #23, once every candidate has been executed from the shared snapshot.
        # It stays in the artifact from here on: whether a pool contained a success at all is
        # a fact about generation, and a selector must never be blamed or credited for it.
        "pool_has_success": None,
        "summary": {"accepted": len(accepted), "rejected": len(rejected),
                    "attempted": len(accepted) + len(rejected),
                    "short_of_requested": max(0, size - len(accepted)) if kind == "natural" else 0,
                    "unique_programs": len(unique),
                    "duplicate_fraction": round(1 - len(unique) / max(1, len(accepted)), 3),
                    # Generation outcomes, reported here so they are never read as selector quality.
                    "aborting_candidates": sum(1 for c in accepted if c.aborts),
                    "acting_candidates": sum(1 for c in accepted if not c.aborts),
                    "strategies": sorted({c.strategy for c in accepted}),
                    "cost_usd": round(sum((c.generation.get("cost_usd") or 0.0) for c in accepted)
                                      + sum((r.generation.get("cost_usd") or 0.0) for r in rejected), 4),
                    "reject_reasons": {stage: sum(1 for r in rejected if r.stage.startswith(stage))
                                       for stage in ("generation", "schema", "grounding", "reachability")}},
    }
    if scene_record:
        pool["scene"]["suite"] = {
            "scenario_id": scene_record["scenario_id"],
            "manifest_lock_sha256": manifest_lock,
            "program_template_id": scene_record["program_template_id"],
            "perturbation_group": scene_record["perturbation_group"],
            "outcome_slice": scene_record["outcome_slice"],
            "observability": scene_record["observability"],
            "rendered_observation": scene_record["rendered_observation"],
            "perception_error_mm": scene_record["perception_error_mm"],
            "scene_spec": {key: scene_record[key] for key in (
                "scenario_id", "split", "scene_seed", "perturbation_seed", "pair_id", "variant",
                "program_pool_id", "program_template_id", "perturbation_group", "outcome_slice",
                "observability", "expected_effect", "scene_parameters")},
        }
    return pool


# --------------------------------------------------------------------------- integrity


def check_pool(pool: dict) -> list[str]:
    """The checks a downstream benchmark is entitled to assume. Empty list means the pool is usable."""
    problems = []
    candidates = pool["candidates"]
    ids = [c["candidate_id"] for c in candidates]
    if len(set(ids)) != len(ids):
        problems.append("duplicate candidate_id")
    if [c["index"] for c in candidates] != list(range(len(candidates))):
        problems.append("candidate index is not a contiguous ranking from 0")
    if [c["sample_index"] for c in candidates] != sorted(c["sample_index"] for c in candidates):
        problems.append("candidates are not in generation order")

    previous = []
    expected = {str(n) for n in PREFIXES if n <= len(candidates)}
    if set(pool["prefixes"]) != expected:
        problems.append(f"expected prefixes {sorted(expected)}, got {sorted(pool['prefixes'])}")
    for n in sorted(int(key) for key in pool["prefixes"]):
        prefix = pool["prefixes"][str(n)]
        if len(prefix) != n:
            problems.append(f"prefix {n} holds {len(prefix)} candidates")
        if prefix[:len(previous)] != previous:
            problems.append(f"prefix {n} is not nested inside the smaller prefixes")
        if prefix != ids[:n]:
            problems.append(f"prefix {n} does not match the pool ordering")
        previous = prefix

    observation_id = pool["scene"]["observation_id"]
    for candidate in candidates:
        for key in ("program", "grounded_trace", "dedup_key", "validation", "candidate_id"):
            if key not in candidate:
                problems.append(f"{candidate.get('candidate_id')}: missing {key}")
        if candidate["program"].get("schema_version") != SCHEMA_VERSION:
            problems.append(f"{candidate['candidate_id']}: wrong program schema version")
        if not candidate["grounded_trace"] and not candidate.get("aborts"):
            problems.append(f"{candidate['candidate_id']}: empty grounded trace")
        if candidate.get("aborts") and candidate["grounded_trace"]:
            problems.append(f"{candidate['candidate_id']}: a declining candidate must not carry a trace")
        leaked = json.dumps(candidate["program"])
        for name, point in pool["scene"]["hidden_truth"].items():
            if f"{point[0]:.4f}" in leaked:
                problems.append(f"{candidate['candidate_id']}: program contains {name} ground truth")
    if pool["kind"] == "natural" and pool["protocol"]["generator"]["model"] == "scripted":
        problems.append("a natural pool must be generated by Claude")
    generator = pool["protocol"].get("generator", {})
    if pool["protocol"].get("generator_hash") != generator_hash(generator):
        problems.append("generator hash does not match the recorded settings")
    if pool["kind"] == "diagnostic":
        named = [c["diagnostic"] for c in candidates]
        if len(set(named)) != len(named):
            problems.append("diagnostic pool repeats a named strategy or fault")
        if any(c["diagnostic_kind"] not in ("strategy", "fault") for c in candidates):
            problems.append("every diagnostic must be labelled a strategy or a fault")
    if not observation_id:
        problems.append("missing observation id")
    if "pool_has_success" not in pool:
        problems.append("pool_has_success was dropped; generation coverage must survive downstream")
    return problems


def validate_root(root: Path) -> int:
    """Check every cached pool under `root`; return the number of problems found."""
    total, files = 0, sorted(root.rglob("*.json"))
    files = [path for path in files if path.name != "index.json"]
    for path in files:
        problems = check_pool(json.loads(path.read_text()))
        total += len(problems)
        status = "ok" if not problems else "; ".join(problems[:4])
        print(f"{'PASS' if not problems else 'FAIL'} {path}: {status}")
    print(f"\n{len(files)} pools checked, {total} problems")
    return total


# --------------------------------------------------------------------------- cli


def parse_seeds(text: str) -> list[int]:
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part.lstrip("-"):
            low, high = part.split("-")
            seeds.extend(range(int(low), int(high) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=tuple(SPLITS), default="test",
                    help="which disjoint seed range to draw scenes from when --seeds is not given")
    ap.add_argument("--seeds", help="explicit scene seeds, e.g. 0,1,2 or 100-107")
    ap.add_argument("--scenes", type=int, default=8, help="how many seeds of the split to use")
    ap.add_argument("--kind", choices=("natural", "diagnostic", "both"), default="both")
    ap.add_argument("--pool-size", type=int, default=POOL_SIZE)
    ap.add_argument("--instruction", default="pick up the {object} and put it on the {destination}")
    ap.add_argument("--object", default="red block")
    ap.add_argument("--destination", default="green pad")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=8, help="concurrent claude CLI calls")
    ap.add_argument("--oversample", type=float, default=1.5,
                    help="attempt budget as a multiple of the pool size, for rejected samples")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--regenerate", action="store_true", help="ignore the cache and sample again")
    ap.add_argument("--allow-dirty", action="store_true", help="allow exploratory generation from tracked edits")
    ap.add_argument("--validate", type=Path, help="check cached pools under this directory and exit")
    ap.add_argument("--scene-manifest", type=Path,
                    help="locked perception-scene manifest from issue #25")
    args = ap.parse_args()

    if args.validate is not None:
        raise SystemExit(1 if validate_root(args.validate) else 0)
    if git_dirty() and not args.allow_dirty:
        raise SystemExit("refusing to generate pools from a dirty tracked worktree; commit the locked code "
                         "or pass --allow-dirty for an explicitly exploratory pool")

    scene_records, manifest_lock = {}, None
    if args.scene_manifest:
        from waddle_wm.perception_scenes import check_manifest
        manifest = json.loads(args.scene_manifest.read_text())
        problems = check_manifest(manifest, args.scene_manifest.parent)
        if problems:
            raise SystemExit("invalid scene manifest: " + "; ".join(problems[:4]))
        if manifest["split"] != args.split:
            raise SystemExit(f"manifest split {manifest['split']!r} does not match --split {args.split!r}")
        scene_records = {row["scene_seed"]: row for row in manifest["scenarios"]}
        manifest_lock = manifest["lock_sha256"]
    seeds = (sorted(scene_records) if scene_records else
             (parse_seeds(args.seeds) if args.seeds else list(SPLITS[args.split])[:args.scenes]))
    kinds = ("natural", "diagnostic") if args.kind == "both" else (args.kind,)
    instruction = args.instruction.format(object=args.object, destination=args.destination)
    index_path = args.root / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    for seed in seeds:
        scene_obj = None
        for kind in kinds:
            previous = None
            directory = args.root / kind / f"{args.object.replace(' ', '_')}_to_{args.destination.replace(' ', '_')}"
            record = scene_records.get(seed)
            filename = f"{record['scenario_id']}.json" if record else f"{args.split}-seed{seed:04d}.json"
            path = directory / filename
            if path.exists() and not args.regenerate:
                cached = json.loads(path.read_text())
                wanted = generator_hash(generator_settings(kind, args.model, args.pool_size, instruction,
                                                            args.workers, args.oversample, args.timeout))
                if cached["protocol"].get("generator_hash") != wanted:
                    print(f"stale   {path}  (generator settings changed since it was cached; "
                          f"regenerating)")
                elif cached["protocol"]["generator"]["pool_size"] < args.pool_size:
                    print(f"small   {path}  ({cached['summary']['accepted']} candidates, "
                          f"{args.pool_size} wanted; extending)")
                    previous = cached
                else:
                    print(f"cached  {path}  ({cached['summary']['accepted']} candidates)")
                    continue
            if scene_obj is None:
                scene_obj = Scene(seed, record)
            pool = build_pool(scene_obj, kind, instruction, args.object, args.destination,
                              args.pool_size, args.model, args.split, args.workers,
                              args.oversample, args.timeout, previous, record, manifest_lock)
            problems = check_pool(pool)
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pool, indent=2, default=float))
            # Keyed by path, not pool id: regenerating under changed settings replaces the
            # entry rather than leaving a dead id behind claiming a pool that no longer exists.
            index[str(path)] = {"pool_id": pool["pool_id"], "kind": kind, "split": args.split,
                                "seed": seed, "observation_id": pool["scene"]["observation_id"],
                                **pool["summary"]}
            print(f"{'wrote  ' if not problems else 'PROBLEM'} {path}  "
                  f"{json.dumps(pool['summary'], default=float)}")
            if pool["summary"]["short_of_requested"]:
                print(f"  ! only {pool['summary']['accepted']} of {args.pool_size} candidates were "
                      f"accepted within the {args.oversample}x attempt budget; raise --oversample")
            for problem in problems:
                print(f"  ! {problem}")
        if scene_obj is not None:
            scene_obj.close()

    args.root.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, default=float))
    print(f"\nindex {index_path} ({len(index)} pools)")


if __name__ == "__main__":
    main()
