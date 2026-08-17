"""Contract checks for the sampled demo flaws.

    uv run python -m waddle_wm.test_chaos          # offline: sampling, plans, compound warps
    uv run python -m waddle_wm.test_chaos --sim    # also boots MuJoCo and checks the pinned scene

The offline checks are the ones that matter for the demo's claim: every draw the sampler can
produce is already outside the tolerances the unchecked arm has to miss, and the plan built from
it is one the verifier and the simulator both accept. `--sim` additionally proves the scene
survives the resets `SkillAgent.observe` performs between arms, which is the failure mode that
would silently turn a scene challenge into an ordinary run.
"""
from __future__ import annotations

import argparse

import numpy as np

from waddle_wm import chaos
from waddle_wm.planner import PICK_PLACE_SHAPE

DETECTIONS = {"red block": [0.42, -0.20, 0.018],
              "blue block": [0.50, -0.18, 0.018],
              "yellow block": [0.64, -0.18, 0.018]}
PAD_XY = (0.50, 0.30)


def check_draws(count: int = 400):
    """Every sampled draw must be one the unchecked baseline is expected to miss."""
    kinds = {}
    for seed in range(count):
        draw = chaos.sample(np.random.default_rng(seed), DETECTIONS,
                            "pick up the red block and put it on the green pad", seed=seed)
        kinds[draw.id] = kinds.get(draw.id, 0) + 1
        assert draw.label, draw.id
        assert chaos.guarantee_fail(draw), f"{draw.id}: {draw.label} is inside every tolerance"
        if draw.id in ("random_grasp", "perception_lie"):
            assert 0.035 <= draw.grasp_magnitude <= 0.070, draw.summary()
            assert draw.grasp_magnitude > chaos.GRASP_TOLERANCE, draw.summary()
        if draw.id == "random_place":
            assert 0.12 <= draw.place_magnitude <= 0.22, draw.summary()
            assert draw.place_magnitude > chaos.PAD_RADIUS, draw.summary()
        if draw.id in ("toward_neighbor", "scene_only") and draw.scene_kind == "neighbor_crowd":
            # The whole point of the crowded case: the offset stays *inside* the coordinate
            # tolerance, so only the scene distinguishes a survivable grasp from a fatal one.
            assert draw.grasp_magnitude <= chaos.GRASP_TOLERANCE, draw.summary()
            assert 0.075 <= draw.scene["spacing"] <= 0.100, draw.summary()
        if draw.id == "wrong_object":
            assert draw.grasp_from != "red block", draw.summary()
        if draw.id == "stale_grasp":
            assert draw.scene["blocks"]["red_block"][:2] != DETECTIONS["red block"][:2]
        assert draw.scene_kind is None or draw.scene_kind in chaos.SCENE_KINDS + ("prior_action",)
    assert len(kinds) >= 5, f"the sampler collapsed onto {sorted(kinds)}"
    print(f"draws: {count} samples, all out of tolerance, kinds {dict(sorted(kinds.items()))}")


def check_opening_plan():
    """The built plan must be executable, pick-and-place shaped, and offset by exactly the draw."""
    for seed in range(60):
        draw = chaos.sample(np.random.default_rng(seed), DETECTIONS,
                            "put the red block on the green pad", seed=seed)
        plan, draw = chaos.build_opening_plan("put the red block on the green pad",
                                              DETECTIONS, PAD_XY, draw)
        assert plan.executable and plan.pick_place_shaped, draw.summary()
        step = plan.steps[0]
        assert tuple(entry["phase"] for entry in step.trace) == PICK_PLACE_SHAPE
        grasp = next(e["target"] for e in step.trace if e["phase"] == "descend")
        place = next(e["target"] for e in step.trace if e["phase"] == "place")
        source = DETECTIONS[draw.grasp_from]
        assert abs(grasp[0] - source[0] - draw.grasp_offset[0]) < 1e-6, draw.summary()
        assert abs(grasp[1] - source[1] - draw.grasp_offset[1]) < 1e-6, draw.summary()
        assert abs(place[0] - PAD_XY[0] - draw.place_offset[0]) < 1e-6, draw.summary()
        assert plan.note == draw.label
    print("plans: 60 sampled draws all build executable, correctly offset pick-and-place plans")


def check_compound():
    """Compound instructions are recognised, and a warp touches exactly one step."""
    for text in ("put the red block on the pad then stack the blue one on it",
                 "move the blue block and the yellow block onto the green pad",
                 "stack the blocks"):
        assert chaos.is_compound(text), text
    for text in ("pick up the red block and put it on the green pad",
                 "place the red block on the green pad", ""):
        assert not chaos.is_compound(text), text

    from waddle_wm.planner import validate
    honest = validate({"intent": "two moves", "action": "execute", "note": "clean",
                       "steps": [{"object": "red block", "destination": "green pad",
                                  "trace": chaos.pick_place_trace((0.42, -0.20), (0.46, 0.30))},
                                 {"object": "blue block", "destination": "green pad",
                                  "trace": chaos.pick_place_trace((0.50, -0.18), (0.54, 0.30))}]})
    draw = chaos.ChaosDraw("random_grasp", "grasp 5.0 cm off", grasp_offset=(0.05, 0.0),
                           target_step_index=0)
    warped, draw = chaos.warp_steps("two moves", honest.steps, draw)
    assert draw.target_step_index == 0
    moved = [index for index, (before, after) in enumerate(zip(honest.steps, warped.steps))
             if before.summary() != after.summary()]
    assert moved == [0], moved
    grasp = next(e["target"] for e in warped.steps[0].trace if e["phase"] == "descend")
    assert abs(grasp[0] - 0.42 - 0.05) < 1e-6
    # The untouched step keeps Claude's own waypoints, byte for byte.
    assert warped.steps[1].summary() == honest.steps[1].summary()

    # A place offset drawn off the pad can point straight out of the workspace; the warp has to
    # stay inside it, because `validate` would otherwise throw the plan away before any arm ran.
    edge = chaos.ChaosDraw("random_place", "release 22 cm off", place_offset=(0.0, 0.22),
                           target_step_index=1)
    fitted, edge = chaos.warp_steps("two moves", honest.steps, edge)
    place = next(e["target"] for e in fitted.steps[1].trace if e["phase"] == "place")
    assert chaos.in_workspace(place[:2]), place
    assert edge.place_magnitude > 0, edge.summary()
    print("compound: instructions classified, one step warped, and the warp stays in the workspace")


def check_pinned_scene():
    """A pinned scene must survive the resets `SkillAgent.observe` performs between arms."""
    import mujoco

    from waddle_wm.sim.env import TabletopEnv

    env = TabletopEnv(seed=0)
    block_xy = [0.42, -0.20]
    draw = chaos.ChaosDraw(
        "scene_only", "occluded and crowded",
        scene={"kind": "neighbor_crowd", "label": "crowded",
               "blocks": {"blue_block": [0.49, -0.20, 0.018],
                          "yellow_block": [0.66, -0.30, 0.018]},
               "occluder": [0.48, -0.26, 0.05],
               "obstacle": [0.46, -0.06, 0.125],
               "friction": [0.30, 0.02, 0.004], "gripper_force": 0.24,
               "camera_offset": [0.006, -0.004, 0.0]})
    baseline_camera = env.model.cam_pos[env.model.camera("demo").id].copy()
    saved = chaos.apply_chaos_scene(env, draw, block_xy)
    obstacle = env.model.body("scene_obstacle_body").mocapid[0]
    occluder = env.model.geom("scene_occluder").id

    for reset_index in range(2):
        env.reset(block_xy)
        blue = env.data.joint("blue_block_free").qpos[:2]
        assert np.allclose(blue, [0.49, -0.20], atol=1e-6), (reset_index, blue)
        assert env.model.geom_pos[occluder][2] > 0, reset_index
        assert env.data.mocap_pos[obstacle][2] > 0, reset_index
        assert env.model.geom_friction[env.model.geom("red_block_geom").id][0] == 0.30
        # The confirmation re-run happens after several resets, so a weakened gripper has to still
        # be weak by then — otherwise the "fresh scene" run would quietly be an easier task.
        assert env.model.actuator_forcerange[
            env.model.actuator(chaos.scene.GRIPPER_ACTUATOR).id][1] == 0.24, reset_index
        assert np.allclose(env.model.cam_pos[env.model.camera("demo").id],
                           baseline_camera + [0.006, -0.004, 0.0], atol=1e-9)

    chaos.restore_chaos_scene(env, saved)
    env.reset(block_xy)
    assert env.model.geom_pos[occluder][2] < 0, "the occluder outlived its run"
    assert env.data.mocap_pos[obstacle][2] < 0, "the obstacle outlived its run"
    assert np.allclose(env.model.cam_pos[env.model.camera("demo").id], baseline_camera, atol=1e-9)
    assert env.model.geom_friction[env.model.geom("red_block_geom").id][0] == 0.8
    assert env.model.actuator_forcerange[env.model.actuator(chaos.scene.GRIPPER_ACTUATOR).id][1] == 5.0
    blue = env.data.joint("blue_block_free").qpos[:2]
    assert np.allclose(blue, [0.50, -0.18], atol=1e-6), blue
    mujoco.mj_forward(env.model, env.data)
    print("scene: the pinned challenge survives two resets and is fully undone by the restore")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sim", action="store_true", help="also boot MuJoCo and check the pinned scene")
    args = ap.parse_args()

    check_draws()
    check_opening_plan()
    check_compound()
    if args.sim:
        check_pinned_scene()
    print("chaos checks passed")


if __name__ == "__main__":
    main()
