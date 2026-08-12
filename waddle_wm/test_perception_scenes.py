"""Small contract check for issue #25. Run with:

    uv run python -m waddle_wm.test_perception_scenes
"""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from waddle_wm import perception_scenes as scenes
from waddle_wm import program as prog
from waddle_wm.pools import Scene, build_pool


def main() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        manifests = [scenes.manifest(split, root) for split in scenes.SPLIT_SEEDS]
        assert not [problem for data in manifests for problem in scenes.check_manifest(data, root)]
        assert scenes.check_suite(manifests) == []
        record = manifests[-1]["scenarios"][0]
        inputs = scenes.selector_inputs(record)
        assert inputs["estimated_state"]["image_derived_estimates"] == \
            inputs["visual"]["image_derived_estimates"]
        assert "raw_frame" not in inputs["estimated_state"] and "raw_frame" in inputs["visual"]
        diagnostic_size = len(prog.diagnostic_programs())
        scene_obj = Scene(record["scene_seed"], record)
        pool = build_pool(scene_obj, "diagnostic", "put the red block on the green pad",
                          "red block", "green pad", diagnostic_size, "scripted", "test", 1, 1.0, 1.0,
                          scene_record=record, manifest_lock=manifests[-1]["lock_sha256"])
        scene_obj.close()
        assert pool["pool_id"] == record["program_pool_id"] + "-diagnostic"
        assert pool["scene"]["suite"]["scene_spec"]["scene_parameters"] == record["scene_parameters"]
        replay = Scene(record["scene_seed"], pool["scene"]["suite"]["scene_spec"])
        assert replay.observation.observation_id == pool["scene"]["observation_id"]
        replay.close()
        occluded = next(row for row in manifests[-1]["scenarios"]
                        if row["outcome_slice"] == "occlusion" and row["variant"] == "challenge")
        occluded_scene = Scene(occluded["scene_seed"], occluded)
        occluded_pool = build_pool(occluded_scene, "diagnostic", "put the red block on the green pad",
                                   "red block", "green pad", diagnostic_size, "scripted", "test", 1, 1.0, 1.0)
        occluded_scene.close()
        assert len(occluded_pool["candidates"]) == diagnostic_size, occluded_pool["rejected"]
        broken = deepcopy(manifests[-1])
        broken["scenarios"][1]["scene_seed"] = broken["scenarios"][0]["scene_seed"]
        assert "duplicate scene_seed" in scenes.check_manifest(broken)
        print("perception scene contract passed: locked splits, paired coverage, selector boundary")


if __name__ == "__main__":
    main()
