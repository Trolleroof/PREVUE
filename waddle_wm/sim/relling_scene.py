"""UR5e + Robotiq 2F-85 industrial pick-and-place scene."""

from pathlib import Path

import mujoco

ROOT = Path(__file__).parents[1] / "assets"
MENAGERIE = ROOT

BLOCK_HALF = 0.018
BLOCK_NAMES = ("red_block", "blue_block", "yellow_block")
BLOCK_SPAWN = ((0.36, -0.18), (0.50, -0.18), (0.64, -0.18))
TARGET_POS = (0.50, 0.30, 0.0)
ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
GRIPPER_ACTUATOR = "2f85/fingers_actuator"


def make_model() -> mujoco.MjModel:
    arm = mujoco.MjSpec.from_file(str(MENAGERIE / "ur5e.xml"))
    gripper = mujoco.MjSpec.from_file(str(MENAGERIE / "2f85.xml"))
    arm.option.impratio = gripper.option.impratio
    arm.option.cone = gripper.option.cone
    arm.site("attachment_site").attach_body(gripper.body("base_mount"), "2f85/", "")

    for key in list(arm.keys):
        arm.delete(key)

    arm.visual.global_.offwidth = arm.visual.global_.offheight = 1024  # room for a UI-sized renderer
    arm.option.timestep = 0.001
    arm.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world = arm.worldbody
    world.add_light(name="key", pos=[0.2, -0.8, 1.5], dir=[0.0, 0.25, -1.0],
                    diffuse=[1.0, 0.94, 0.84], specular=[0.3, 0.3, 0.3])
    world.add_light(name="fill", pos=[-0.8, 0.1, 1.0], dir=[0.65, -0.1, -0.8],
                    diffuse=[0.35, 0.5, 0.8])
    world.add_light(name="rim", pos=[0.8, 0.8, 1.4], dir=[-0.35, -0.45, -0.8],
                    diffuse=[0.55, 0.65, 1.0])
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[0.95, 0.8, 0.05], rgba=[0.06, 0.13, 0.22, 1.0])
    for x in (-0.7, -0.35, 0.0, 0.35, 0.7):
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, 0, 0.002],
                       size=[0.003, 0.8, 0.002], rgba=[0.25, 0.38, 0.52, 1.0],
                       contype=0, conaffinity=0)
    for y in (-0.6, -0.3, 0.0, 0.3, 0.6):
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, y, 0.003],
                       size=[0.95, 0.003, 0.002], rgba=[0.25, 0.38, 0.52, 1.0],
                       contype=0, conaffinity=0)
    world.add_site(name="target", pos=list(TARGET_POS),
                   type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.105, 0.002, 0],
                   rgba=[0.08, 0.8, 0.25, 0.75])
    # Scene-suite props live below the table unless a locked scene enables them. Keeping
    # them in the one model lets every candidate restore the same MuJoCo state.
    obstacle = world.add_body(name="scene_obstacle_body", pos=[0, 0, -1], mocap=True)
    obstacle.add_geom(name="scene_obstacle", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.018, 0.025, 0.125], rgba=[0.7, 0.72, 0.76, 1],
                      contype=1, conaffinity=1)
    world.add_geom(name="scene_occluder", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0, 0, -1], size=[0.020, 0.020, 0.05], rgba=[0.12, 0.12, 0.12, 1],
                   contype=0, conaffinity=0)
    for name, (x, y) in zip(BLOCK_NAMES, BLOCK_SPAWN):
        body = world.add_body(name=name, pos=[x, y, BLOCK_HALF])
        body.add_freejoint(name=f"{name}_free")
        body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[BLOCK_HALF] * 3, mass=0.025,
                      rgba={"red_block": [0.9, 0.05, 0.05, 1],
                            "blue_block": [0.04, 0.25, 0.9, 1],
                            "yellow_block": [0.95, 0.72, 0.03, 1]}[name],
                      friction=[0.8, 0.03, 0.01], condim=6)
    world.add_camera(name="demo", pos=[1.15, -1.55, 0.92],
                     xyaxes=[0.78, 0.62, 0, -0.27, 0.34, 0.90])
    return arm.compile()


def reset(model, data):
    mujoco.mj_resetData(model, data)
    for joint, q in zip(ARM_JOINTS, (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)):
        data.joint(joint).qpos[0] = q
        data.actuator(joint.removesuffix("_joint")).ctrl[0] = q
    for name, (x, y) in zip(BLOCK_NAMES, BLOCK_SPAWN):
        data.joint(f"{name}_free").qpos[:] = [x, y, BLOCK_HALF, 1, 0, 0, 0]
    data.actuator(GRIPPER_ACTUATOR).ctrl[0] = 0
    mujoco.mj_forward(model, data)
