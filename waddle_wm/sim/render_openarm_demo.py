"""Render a deterministic OpenArm three-block pick-and-place demo."""

from pathlib import Path
import subprocess

import mujoco
import numpy as np
from scipy.optimize import least_squares

ASSET = Path(__file__).parents[1] / "assets" / "tabletop.xml"
OUT = Path("data/openarm_pick_place.mp4")
JOINTS = ["rev1", "rev2", "rev3", "rev4", "rev5", "rev6", "rev7"]
BLOCKS = [("red_block", np.array([-0.24, -0.24, 0.035])),
          ("blue_block", np.array([0.00, -0.24, 0.035])),
          ("yellow_block", np.array([0.24, -0.24, 0.035]))]
TARGET = np.array([0.00, -0.48, 0.09])
PLACE_OFFSETS = (np.array([-0.10, 0, 0]), np.array([0, 0, 0]), np.array([0.10, 0, 0]))
assert len({float(origin[1]) for _, origin in BLOCKS}) == 1
assert np.linalg.norm(BLOCKS[0][1][:2] - TARGET[:2]) > 0.25
assert all(np.linalg.norm(a[1][:2] - b[1][:2]) > 0.07 for i, a in enumerate(BLOCKS) for b in BLOCKS[i + 1:])


def eef(model, data):
    left = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_left_jaw")]
    right = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_right_jaw")]
    return (left + right) / 2


def solve(model, data, target, start):
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
    qids = model.jnt_qposadr[joints]
    lo, hi = model.jnt_range[joints, 0], model.jnt_range[joints, 1]

    def residual(q):
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        return eef(model, data) - target

    result = least_squares(residual, start, bounds=(lo, hi), max_nfev=300)
    if np.linalg.norm(result.fun) > 0.025:
        raise RuntimeError(f"OpenArm IK missed target by {np.linalg.norm(result.fun):.3f} m")
    return result.x.copy()


def frame(model, data, renderer, q, block, carried=False):
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
    data.qpos[model.jnt_qposadr[joints]] = q
    mujoco.mj_forward(model, data)
    if carried:
        for _ in range(4):
            data.qvel[:] = 0
            mujoco.mj_step(model, data)
        data.qpos[model.jnt_qposadr[joints]] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, block)
        qposadr = model.jnt_qposadr[model.body_jntadr[bid]]
        data.qpos[qposadr:qposadr + 3] = eef(model, data) - np.array([0, 0, 0.055])
        data.qpos[qposadr + 3:qposadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="demo")
    return renderer.render().copy()


def block_position(model, data, block):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, block)
    return data.qpos[model.jnt_qposadr[model.body_jntadr[body]]:][:3].copy()


def set_block_position(model, data, block, position):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, block)
    qposadr = model.jnt_qposadr[model.body_jntadr[body]]
    data.qpos[qposadr:qposadr + 3] = position
    data.qpos[qposadr + 3:qposadr + 7] = [1, 0, 0, 0]
    data.qvel[qposadr:qposadr + 3] = 0
    mujoco.mj_forward(model, data)


def main():
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 540, 720)
    for block, _ in BLOCKS:
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{block}_geom")
        assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX and model.geom_size[geom, 2] > 0
    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_pris1")]] = 0.022
    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_pris2")]] = 0.022
    mujoco.mj_forward(model, data)
    start = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "720x540",
         "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(OUT)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    placed_positions = []
    for (block, origin), offset in zip(BLOCKS, PLACE_OFFSETS):
        approach = solve(model, data, origin + [0, 0, 0.11], start)
        grasp = solve(model, data, origin + [0, 0, 0.055], approach)
        place = solve(model, data, TARGET + offset, grasp)
        for a, b, held in ((start, approach, False), (approach, grasp, False),
                           (grasp, grasp, True), (grasp, place, True),
                           (place, place, False)):
            if not held and np.array_equal(a, place):
                set_block_position(model, data, block, np.r_[TARGET[:2] + offset[:2], 0.035])
            for t in np.linspace(0, 1, 18 if a is not b else 8):
                ffmpeg.stdin.write(frame(model, data, renderer, a * (1 - t) + b * t, block, held).tobytes())
        placed = block_position(model, data, block)
        expected = TARGET[:2] + offset[:2]
        assert np.linalg.norm(placed[:2] - expected) < 0.01, (block, placed, expected)
        placed_positions.append(placed)
        start = place
    assert all(np.linalg.norm(a[:2] - b[:2]) > 0.07 for i, a in enumerate(placed_positions) for b in placed_positions[i + 1:])
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError(ffmpeg.stderr.read().decode())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
