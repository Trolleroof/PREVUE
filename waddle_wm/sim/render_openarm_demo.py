"""Render a physical UR5e/Robotiq top-down block pick-and-place demo."""

from pathlib import Path
import subprocess

import mujoco
import numpy as np

from waddle_wm.sim import relling_scene as scene

OUT = Path("data/ur5e_pick_place.mp4")
APPROACH_DOWN = np.array([0.0, 0.0, -1.0])
HOVER_Z = 0.24
GRASP_Z = 0.015
TRANSIT_Z = 0.30
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 255.0


def arm_qpos(model, data):
    return np.array([data.joint(j).qpos[0] for j in scene.ARM_JOINTS])


def pinch_pos(model, data):
    return data.site_xpos[model.site("2f85/pinch").id].copy()


def solve_ik(model, data, target, q_init):
    target = np.asarray(target, dtype=float)
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    qadr = np.array([model.joint(j).qposadr[0] for j in scene.ARM_JOINTS])
    dof = np.array([model.joint(j).dofadr[0] for j in scene.ARM_JOINTS])
    scratch.qpos[qadr] = q_init
    site_id = model.site("2f85/pinch").id
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(220):
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        pos = scratch.site_xpos[site_id]
        z_axis = scratch.site_xmat[site_id].reshape(3, 3)[:, 2]
        err = np.r_[target - pos, np.cross(z_axis, APPROACH_DOWN)]
        if np.linalg.norm(err[:3]) < 0.0015 and np.linalg.norm(err[3:]) < 0.03:
            return scratch.qpos[qadr].copy()
        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        J = np.vstack((jacp[:, dof], jacr[:, dof]))
        dq = J.T @ np.linalg.solve(J @ J.T + 0.08**2 * np.eye(6), err)
        dq *= min(1.0, 0.14 / max(np.max(np.abs(dq)), 1e-9))
        scratch.qpos[qadr] += dq
        for j, adr in zip(scene.ARM_JOINTS, qadr):
            if model.joint(j).limited:
                scratch.qpos[adr] = np.clip(scratch.qpos[adr], *model.joint(j).range)
    raise RuntimeError(f"IK failed for {target}: error={np.linalg.norm(err[:3]):.4f}m")


def move(model, data, target_q, gripper, renderer, ffmpeg, frames, on_hold=False):
    start = np.array([data.actuator(j.removesuffix("_joint")).ctrl[0]
                      for j in scene.ARM_JOINTS])
    steps = max(12, int(np.max(np.abs(target_q - start)) / 0.025))
    for q in np.linspace(start, target_q, steps):
        for j, value in zip(scene.ARM_JOINTS, q):
            data.actuator(j.removesuffix("_joint")).ctrl[0] = value
        data.actuator(scene.GRIPPER_ACTUATOR).ctrl[0] = gripper
        for _ in range(4):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera="demo")
        frames.append(renderer.render().copy())
    # The take-home controller settles the position actuators under load; a waypoint
    # is not considered reached just because the command was written.
    for i in range(int(0.45 / model.opt.timestep)):
        data.actuator(scene.GRIPPER_ACTUATOR).ctrl[0] = gripper
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            renderer.update_scene(data, camera="demo")
            frames.append(renderer.render().copy())
    return on_hold


def settle_gripper(model, data, gripper, renderer, frames, seconds=0.8):
    data.actuator(scene.GRIPPER_ACTUATOR).ctrl[0] = gripper
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if _ % 10 == 0:
            renderer.update_scene(data, camera="demo")
            frames.append(renderer.render().copy())


def block_z(model, data, name):
    return float(data.joint(f"{name}_free").qpos[2])


def main():
    model = scene.make_model()
    data = mujoco.MjData(model)
    scene.reset(model, data)
    renderer = mujoco.Renderer(model, 480, 640)
    frames = []
    start = arm_qpos(model, data)
    max_lift = 0.0
    placed = []

    assert model.neq == 3, "Robotiq linkage equalities only; no object welds"
    assert all(model.geom_type[model.geom(f"{n}_geom").id] == mujoco.mjtGeom.mjGEOM_BOX
               for n in scene.BLOCK_NAMES)

    for slot, (name, (x, y)) in enumerate(zip(scene.BLOCK_NAMES, scene.BLOCK_SPAWN)):
        perceived = np.array([x, y, scene.BLOCK_HALF])
        target_xy = np.array(scene.TARGET_POS[:2]) + np.array([-0.055 + slot * 0.055, 0.0])
        # Every pick and place uses the same vertical top-down approach.
        waypoints = (
            (np.r_[perceived[:2], HOVER_Z], False),
            (np.r_[perceived[:2], GRASP_Z], False),
        )
        q = start
        for point, _ in waypoints:
            q = solve_ik(model, data, point, q)
            move(model, data, q, GRIPPER_OPEN, renderer, None, frames)
        settle_gripper(model, data, GRIPPER_CLOSED, renderer, frames)
        lift_q = solve_ik(model, data, np.r_[perceived[:2], HOVER_Z], q)
        move(model, data, lift_q, GRIPPER_CLOSED, renderer, None, frames, True)
        max_lift = max(max_lift, block_z(model, data, name))
        place_high = solve_ik(model, data, np.r_[target_xy, TRANSIT_Z], lift_q)
        move(model, data, place_high, GRIPPER_CLOSED, renderer, None, frames, True)
        place = solve_ik(model, data, np.r_[target_xy, GRASP_Z], place_high)
        move(model, data, place, GRIPPER_CLOSED, renderer, None, frames, True)
        settle_gripper(model, data, GRIPPER_OPEN, renderer, frames, 0.5)
        retreat = solve_ik(model, data, np.r_[target_xy, HOVER_Z], place)
        move(model, data, retreat, GRIPPER_OPEN, renderer, None, frames)
        placed.append(data.joint(f"{name}_free").qpos[:3].copy())
        start = retreat

    assert max_lift > 0.09, f"physics never lifted a block: max z={max_lift:.3f}"
    assert all(np.linalg.norm(a[:2] - b[:2]) > 0.035
               for i, a in enumerate(placed) for b in placed[i + 1:])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
         "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(OUT)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    for frame in frames:
        ffmpeg.stdin.write(frame.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError(ffmpeg.stderr.read().decode())
    print(f"wrote {OUT} ({len(frames)} frames, max_lift={max_lift:.3f}m)")


if __name__ == "__main__":
    main()
