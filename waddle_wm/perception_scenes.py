"""Locked perception-dependent MuJoCo scenes consumed by the #18 selector benchmark.

    uv run python -m waddle_wm.perception_scenes --build data/perception_scenes
    uv run python -m waddle_wm.perception_scenes --validate data/perception_scenes
    uv run python -m waddle_wm.perception_scenes --outcome-preflight run.json --pools data/pools

The manifest is the boundary: selector inputs are constructed field-by-field by
``selector_inputs``; MuJoCo truth and outcome slices never cross it.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from collections import Counter
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from waddle_wm.perception import QUERIES, SceneCamera, landing_pad
from waddle_wm.pools import Scene
from waddle_wm.program import SCHEMA_VERSION as PROGRAM_SCHEMA_VERSION
from waddle_wm.sim import relling_scene

SCHEMA_VERSION = 1
SLICES = {
    "grasp_misalignment": "visible_omitted_by_coordinates",
    "occlusion": "visible_omitted_by_coordinates",
    "block_orientation": "visible_omitted_by_coordinates",
    "path_obstruction": "visible_omitted_by_coordinates",
    "slip_drop": "latent_physics",
    "prior_action_state_change": "visible_omitted_by_coordinates",
    "release_dynamics": "latent_physics",
    "malformed_sequence": "plan_visible_control",
    "unreachable_waypoint": "plan_visible_control",
    "obvious_target_miss": "plan_visible_control",
}
VARIANTS = ("baseline", "challenge")
SPLIT_SEEDS = {
    "train": list(range(0, 20)),
    "calibration": list(range(40, 60)),
    "test": list(range(100, 120)),
}
HIDDEN_KEYS = {"hidden_truth", "outcome_slice", "observability", "perturbation_seed",
               "scene_parameters", "expected_effect", "perception_error_mm"}


def _sha(payload) -> str:
    if not isinstance(payload, bytes):
        payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def implementation_hash() -> str:
    functions = (relling_scene.make_model, SceneCamera.bounding_box, SceneCamera.detect_in_base,
                 _parameters, apply_scene_spec, selector_inputs)
    return _sha("\n".join(inspect.getsource(function) for function in functions))


def _parameters(slice_name: str, variant: str, pair_index: int, split_index: int) -> dict:
    challenge = variant == "challenge"
    angle = (pair_index * 31) % 180 - 90
    split_shift = (-0.008, 0.008, 0.0)[split_index]
    base = {
        "block_positions": {
            "red_block": [0.38 + 0.008 * (pair_index % 4) + split_shift,
                          -0.20 + 0.012 * (pair_index % 3) - split_shift / 2, 0.018],
            "blue_block": [0.52 - split_shift / 2, -0.14 + split_shift, 0.018],
            "yellow_block": [0.63 + split_shift / 2, -0.23 - split_shift, 0.018],
        },
        "red_block_size": [0.018, 0.018, 0.018],
        "red_block_yaw_deg": 0.0,
        "obstacle": None,
        "occluder": None,
        "camera_offset": [0.006 * ((pair_index % 3) - 1) + split_shift / 2,
                          0.004 * ((pair_index % 2) * 2 - 1), 0.0],
        "light_scale": 0.9 + 0.05 * (pair_index % 3) + 0.02 * split_index,
        "table_rgba": [0.06 + 0.01 * (pair_index % 2), 0.13 + 0.005 * split_index, 0.22, 1.0],
        "red_block_friction": [0.8, 0.03, 0.01],
        "red_block_mass_kg": 0.025,
        "gripper_force_limit_n": 5.0,
        "contact_solref": [0.02, 1.0],
        "control_delay_steps": 0,
        "prior_action": None,
    }
    if challenge:
        base["table_rgba"][0] += 0.005
    red = base["block_positions"]["red_block"]
    if slice_name == "grasp_misalignment" and challenge:
        base["red_block_size"] = [0.029, 0.012, 0.018]
        base["red_block_yaw_deg"] = 38.0
    elif slice_name == "occlusion" and challenge:
        base["occluder"] = [red[0] + 0.060, red[1] - 0.060, 0.05]
    elif slice_name == "block_orientation":
        base["red_block_size"] = [0.030, 0.011, 0.018]
        base["red_block_yaw_deg"] = angle if challenge else 0.0
    elif slice_name == "path_obstruction" and challenge:
        base["obstacle"] = [0.505, -0.080, 0.125]
    elif slice_name == "slip_drop" and challenge:
        base["red_block_friction"] = [0.30, 0.02, 0.004]
        base["gripper_force_limit_n"] = 0.24
        base["control_delay_steps"] = 1
    elif slice_name == "prior_action_state_change" and challenge:
        base["block_positions"]["red_block"][0] += 0.055
        base["prior_action"] = "red block displaced +55 mm after an earlier failed grasp"
    elif slice_name == "release_dynamics" and challenge:
        base["red_block_mass_kg"] = 0.065
        base["red_block_friction"] = [0.25, 0.015, 0.004]
        base["contact_solref"] = [0.01, 0.1]
        base["control_delay_steps"] = 2
    return base


def scene_specs(split: str) -> list[dict]:
    if split not in SPLIT_SEEDS:
        raise ValueError(f"unknown split {split!r}")
    specs = []
    for index, ((slice_name, observability), variant) in enumerate(
            (item, variant) for item in SLICES.items() for variant in VARIANTS):
        pair_index = index // 2
        seed = SPLIT_SEEDS[split][index]
        scenario_id = f"{split}-{slice_name}-{variant}-v1"
        specs.append({
            "scenario_id": scenario_id,
            "split": split,
            "scene_seed": seed,
            "perturbation_seed": {"train": 1000, "calibration": 2000, "test": 3000}[split] + index,
            "pair_id": f"{split}-{slice_name}-pair-v1",
            "variant": variant,
            "program_pool_id": f"{scenario_id}-pool",
            "program_template_id": f"{split}-{slice_name}-templates-v1",
            "perturbation_group": slice_name,
            "outcome_slice": slice_name,
            "observability": observability,
            "expected_effect": "control program defect" if observability == "plan_visible_control"
                               else ("not expected to be inferable from one initial frame"
                                     if observability == "latent_physics"
                                     else "raw frame retains state omitted by centre coordinates"),
            "scene_parameters": _parameters(slice_name, variant, pair_index,
                                              list(SPLIT_SEEDS).index(split)),
        })
    return specs


def apply_scene_spec(env, spec: dict) -> None:
    """Apply one manifest record to the existing UR5e model, before observation."""
    params = spec["scene_parameters"]
    red_geom = env.model.geom("red_block_geom")
    env.model.geom_size[red_geom.id] = params["red_block_size"]
    env.model.geom_friction[red_geom.id] = params["red_block_friction"]
    env.model.body_mass[env.model.body("red_block").id] = params["red_block_mass_kg"]
    gripper = env.model.actuator(relling_scene.GRIPPER_ACTUATOR).id
    env.model.actuator_forcerange[gripper] = [-params["gripper_force_limit_n"],
                                               params["gripper_force_limit_n"]]
    for name in ("red_block_geom", "table"):
        env.model.geom_solref[env.model.geom(name).id] = params["contact_solref"]
    env.model.geom_pos[env.model.geom("scene_occluder").id] = params["occluder"] or [0.0, 0.0, -1.0]
    camera = env.model.camera("demo").id
    env.model.cam_pos[camera] += params["camera_offset"]
    env.model.light_diffuse[:] *= params["light_scale"]
    env.model.geom_rgba[env.model.geom("table").id] = params["table_rgba"]
    mujoco.mj_setConst(env.model, env.data)
    env.reset(blocks=params["block_positions"])
    obstacle = env.model.body("scene_obstacle_body").mocapid[0]
    env.data.mocap_pos[obstacle] = params["obstacle"] or [0.0, 0.0, -1.0]
    red_joint = env.data.joint("red_block_free").qpos
    yaw = math.radians(params["red_block_yaw_deg"])
    red_joint[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    env.set_control_delay(params["control_delay_steps"])
    mujoco.mj_forward(env.model, env.data)


def _truth(scene_obj: Scene, spec: dict) -> dict:
    env = scene_obj.env
    return {
        "blocks": {name: {"position": env.data.joint(f"{name}_free").qpos[:3].round(6).tolist(),
                           "quaternion": env.data.joint(f"{name}_free").qpos[3:7].round(6).tolist()}
                   for name in relling_scene.BLOCK_NAMES},
        "target_position": env.model.site_pos[env.model.site("target").id].round(6).tolist(),
        "physics": {key: spec["scene_parameters"][key] for key in
                    ("red_block_friction", "red_block_mass_kg", "gripper_force_limit_n",
                     "contact_solref", "control_delay_steps")},
    }


def materialize(spec: dict, root: Path) -> dict:
    scene_obj = Scene(spec["scene_seed"], spec)
    try:
        frame = scene_obj.camera.rgb()
        detections = scene_obj.camera.detect_all(QUERIES)
        relative = Path("observations") / spec["split"] / f"{spec['scenario_id']}.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(path, frame)
        estimates = {d.label: d.summary() for d in detections}
        truth = _truth(scene_obj, spec)
        errors = {}
        for label, detection in estimates.items():
            name = label.replace(" ", "_")
            point = truth["blocks"][name]["position"]
            errors[label] = round(float(np.linalg.norm(
                np.subtract(detection["point_base"][:2], point[:2]))) * 1000, 3)
        return {**spec,
                "rendered_observation": {"path": str(relative), "sha256": _sha(frame.tobytes()),
                                         "shape": list(frame.shape)},
                "image_derived_estimates": estimates,
                "perception_error_mm": errors,
                "landing_pad": {"centre": landing_pad(scene_obj.env.model)[0],
                                "radius": landing_pad(scene_obj.env.model)[1]},
                "hidden_truth": truth}
    finally:
        scene_obj.close()


def selector_inputs(record: dict) -> dict[str, dict]:
    """The exact #18 information boundary; truth stays outside both arms."""
    shared = {"scenario_id": record["scenario_id"], "program_pool_id": record["program_pool_id"],
              "image_derived_estimates": record["image_derived_estimates"],
              "landing_pad": record["landing_pad"]}
    return {"estimated_state": shared, "visual": {**shared, "raw_frame": record["rendered_observation"]}}


def manifest(split: str, root: Path) -> dict:
    records = [materialize(spec, root) for spec in scene_specs(split)]
    body = {"scene_schema_version": SCHEMA_VERSION,
            "protocol": {"program_schema_version": PROGRAM_SCHEMA_VERSION,
                         "implementation_sha256": implementation_hash(),
                         "camera": "demo", "resolution": [256, 256],
                         "detector_queries": list(QUERIES)},
            "split": split,
            "locked": True, "slices": list(SLICES), "scenarios": records}
    return {**body, "lock_sha256": _sha(body)}


def check_manifest(data: dict, root: Path | None = None) -> list[str]:
    problems = []
    body = {key: value for key, value in data.items() if key != "lock_sha256"}
    if data.get("lock_sha256") != _sha(body):
        problems.append("manifest lock hash does not match its contents")
    if data.get("protocol", {}).get("implementation_sha256") != implementation_hash():
        problems.append("scene or perception implementation changed since the manifest was locked")
    records = data.get("scenarios") or []
    for key in ("scenario_id", "scene_seed", "perturbation_seed"):
        values = [row.get(key) for row in records]
        if len(values) != len(set(values)):
            problems.append(f"duplicate {key}")
    observation_hashes = [row.get("rendered_observation", {}).get("sha256") for row in records]
    if len(observation_hashes) != len(set(observation_hashes)):
        problems.append("duplicate rendered observation")
    coverage = Counter(row.get("outcome_slice") for row in records)
    for name in SLICES:
        if coverage[name] == 0:
            problems.append(f"slice {name} has no coverage")
    pairs = Counter(row.get("pair_id") for row in records)
    if any(count != 2 for count in pairs.values()):
        problems.append("every perturbation pair must contain exactly two scenes")
    for row in records:
        inputs = selector_inputs(row)
        if inputs["estimated_state"]["image_derived_estimates"] != inputs["visual"]["image_derived_estimates"]:
            problems.append(f"{row.get('scenario_id')}: selectors received different coordinate estimates")
        if set(_keys_in(inputs["estimated_state"])) & HIDDEN_KEYS or set(_keys_in(inputs["visual"])) & HIDDEN_KEYS:
            problems.append(f"{row.get('scenario_id')}: hidden field entered selector input")
        if root is not None:
            observation = row.get("rendered_observation") or {}
            path = root / observation.get("path", "")
            if not path.is_file():
                problems.append(f"{row.get('scenario_id')}: rendered observation is missing")
            elif _sha(iio.imread(path).tobytes()) != observation.get("sha256"):
                problems.append(f"{row.get('scenario_id')}: rendered observation hash changed")
    return problems


def _keys_in(node) -> list[str]:
    if isinstance(node, dict):
        return [key for name, value in node.items() for key in (name, *_keys_in(value))]
    if isinstance(node, list):
        return [key for item in node for key in _keys_in(item)]
    return []


def check_suite(manifests: list[dict]) -> list[str]:
    problems = []
    for key in ("scene_seed", "perturbation_seed", "scenario_id", "program_template_id"):
        owners = {}
        for data in manifests:
            for row in data.get("scenarios") or []:
                value = row.get(key)
                if value in owners and owners[value] != data.get("split"):
                    problems.append(f"{key} {value!r} leaks across {owners[value]} and {data.get('split')}")
                owners[value] = data.get("split")
    hashes = {}
    for data in manifests:
        for row in data.get("scenarios") or []:
            value = row.get("rendered_observation", {}).get("sha256")
            if value in hashes and hashes[value] != data.get("split"):
                problems.append(f"rendered observation leaks across {hashes[value]} and {data.get('split')}")
            hashes[value] = data.get("split")
    return problems


def outcome_preflight(artifact: dict, pools: list[dict]) -> tuple[dict, list[str]]:
    """Require real diagnostic outcome coverage before #18 may consume the locked test split."""
    from waddle_wm.counterfactual import check_execution, scenario_id_of
    from waddle_wm.program import FAULTS, STRATEGIES

    problems = [problem for problem in check_execution(artifact)
                if "dirty worktree" not in problem]
    by_scenario = {scenario_id_of(pool): pool for pool in pools}
    rows = {}
    for scenario_id, cells in artifact.get("execution", {}).items():
        pool = by_scenario.get(scenario_id)
        if pool is None or not pool.get("scene", {}).get("suite"):
            problems.append(f"{scenario_id}: no locked scene-suite pool")
            continue
        suite = pool["scene"]["suite"]
        cell = next((item for item in cells if item.get("physics_seed") == 0), None)
        if cell is None:
            problems.append(f"{scenario_id}: no canonical physics-seed outcome")
            continue
        names = {candidate["candidate_id"]: candidate.get("diagnostic")
                 for candidate in pool["candidates"]}
        expected = set(STRATEGIES) | set(FAULTS)
        if set(names.values()) != expected:
            problems.append(f"{suite['scenario_id']}: diagnostic strategy coverage changed")
        outcomes = {names[candidate_id]: outcome
                    for candidate_id, outcome in cell["outcomes"].items()}
        successes = sum(bool(outcome["success"]) for outcome in outcomes.values())
        if not 0 < successes < len(outcomes):
            problems.append(f"{suite['scenario_id']}: needs both successful and failed candidates")
        rows[(suite["outcome_slice"], suite["scene_spec"]["variant"])] = outcomes

    report = {}
    for slice_name, observability in SLICES.items():
        baseline = rows.get((slice_name, "baseline"))
        challenge = rows.get((slice_name, "challenge"))
        if baseline is None or challenge is None:
            problems.append(f"{slice_name}: missing baseline/challenge outcome pair")
            continue
        changed = sorted(name for name in baseline
                         if name in challenge and bool(baseline[name]["success"]) !=
                         bool(challenge[name]["success"]))
        report[slice_name] = {
            "baseline_successes": sum(bool(row["success"]) for row in baseline.values()),
            "challenge_successes": sum(bool(row["success"]) for row in challenge.values()),
            "changed_candidates": changed,
        }
        if observability != "plan_visible_control" and not changed:
            problems.append(f"{slice_name}: no candidate outcome changes across the scene pair")
    return report, problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", type=Path)
    group.add_argument("--validate", type=Path)
    group.add_argument("--outcome-preflight", type=Path)
    ap.add_argument("--pools", type=Path, help="diagnostic pools used by --outcome-preflight")
    args = ap.parse_args()
    if args.outcome_preflight:
        if args.pools is None:
            raise SystemExit("--outcome-preflight requires --pools")
        artifact = json.loads(args.outcome_preflight.read_text())
        pools = [json.loads(path.read_text()) for path in sorted(args.pools.rglob("*.json"))
                 if path.name != "index.json"]
        report, problems = outcome_preflight(artifact, pools)
        for slice_name, row in report.items():
            print(f"{slice_name:26} {row['baseline_successes']:2} -> "
                  f"{row['challenge_successes']:2} successes; changed={row['changed_candidates']}")
        print(f"{'PASS' if not problems else 'FAIL'}: {len(report)}/{len(SLICES)} slices, "
              f"{len(problems)} problems")
        for problem in problems:
            print(f"  ! {problem}")
        raise SystemExit(1 if problems else 0)
    root = args.build or args.validate
    paths = [root / f"{split}.json" for split in SPLIT_SEEDS]
    if args.build:
        root.mkdir(parents=True, exist_ok=True)
        for split, path in zip(SPLIT_SEEDS, paths):
            data = manifest(split, root)
            path.write_text(json.dumps(data, indent=2) + "\n")
            print(f"wrote {path}: {len(data['scenarios'])} scenes, {len(data['slices'])} slices")
    loaded = [json.loads(path.read_text()) for path in paths]
    problems = [problem for data in loaded for problem in check_manifest(data, root)] + check_suite(loaded)
    print(f"{'PASS' if not problems else 'FAIL'}: {sum(len(x['scenarios']) for x in loaded)} scenes, "
          f"{len(SLICES)} slices per split, {len(problems)} problems")
    for problem in problems:
        print(f"  ! {problem}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
