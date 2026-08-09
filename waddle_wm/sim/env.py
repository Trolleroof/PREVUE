"""UR5e/Robotiq skill environment used by the dataset generator."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from waddle_wm.sim import relling_scene as scene

SKILLS = ("pick_place",)
APPROACH_DOWN = np.array([0.0, 0.0, -1.0])
HOVER_Z, GRASP_Z, TRANSIT_Z = 0.24, 0.015, 0.30
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 255.0
TARGET_RADIUS = 0.105


@dataclass
class Episode:
    skill: str
    params: dict
    skill_trace: list[dict]
    state_before: dict
    state_after: dict
    success: bool
    failure_mode: str | None
    frames: np.ndarray
    frame_times: list[float] = field(default_factory=list)


class TabletopEnv:
    """One physical UR5e pick-and-place execution, rendered at a fixed rate."""

    def __init__(self, camera="demo", width=256, height=256, fps=10, seed=None):
        self.model = scene.make_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.camera, self.fps = camera, fps
        self.rng = np.random.default_rng(seed)
        self._frame_steps = max(1, round(1 / (fps * self.model.opt.timestep)))
        self._qadr = np.array([self.model.joint(j).qposadr[0] for j in scene.ARM_JOINTS])
        self._dof = np.array([self.model.joint(j).dofadr[0] for j in scene.ARM_JOINTS])
        self._pinch = self.model.site("2f85/pinch").id
        self._frames, self._frame_times = [], []
        self._max_lift = 0.0
        self.reset()

    def reset(self, block_xy=None, target_xy=None):
        scene.reset(self.model, self.data)
        if block_xy is not None:
            self.data.joint("red_block_free").qpos[:2] = block_xy
        if target_xy is not None:
            self.model.site_pos[self.model.site("target").id, :2] = target_xy
        mujoco.mj_forward(self.model, self.data)
        self._frames, self._frame_times = [], []
        self._max_lift = float(self.data.joint("red_block_free").qpos[2])
        return self.state()

    def state(self):
        block = self.data.joint("red_block_free").qpos
        grip = self.data.site("2f85/pinch").xpos
        target = self.model.site_pos[self.model.site("target").id]
        return {
            "block_pos": block[:3].tolist(),
            "block_quat": block[3:7].tolist(),
            "gripper_pos": grip.tolist(),
            "target_pos": target[:2].tolist(),
            "target_distance": float(np.linalg.norm(block[:2] - target[:2])),
            "max_block_z": self._max_lift,
        }

    def sample_scene(self):
        return self.rng.uniform([0.34, -0.22], [0.42, -0.14])

    def _capture(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        self._frames.append(self.renderer.render().copy())
        self._frame_times.append(float(self.data.time))

    def _step(self, gripper, frames=True):
        self.data.actuator(scene.GRIPPER_ACTUATOR).ctrl[0] = gripper
        mujoco.mj_step(self.model, self.data)
        self._max_lift = max(self._max_lift, float(self.data.joint("red_block_free").qpos[2]))
        if frames and round(self.data.time / self.model.opt.timestep) % self._frame_steps == 0:
            self._capture()

    def _ik(self, target, q_init):
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self._qadr] = q_init
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(220):
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)
            pos = scratch.site_xpos[self._pinch]
            z_axis = scratch.site_xmat[self._pinch].reshape(3, 3)[:, 2]
            err = np.r_[np.asarray(target) - pos, np.cross(z_axis, APPROACH_DOWN)]
            if np.linalg.norm(err[:3]) < 0.0015 and np.linalg.norm(err[3:]) < 0.03:
                return scratch.qpos[self._qadr].copy()
            mujoco.mj_jacSite(self.model, scratch, jacp, jacr, self._pinch)
            jac = np.vstack((jacp[:, self._dof], jacr[:, self._dof]))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.08**2 * np.eye(6), err)
            scratch.qpos[self._qadr] += dq * min(1.0, 0.14 / max(np.max(abs(dq)), 1e-9))
            for joint, adr in zip(scene.ARM_JOINTS, self._qadr):
                if self.model.joint(joint).limited:
                    scratch.qpos[adr] = np.clip(scratch.qpos[adr], *self.model.joint(joint).range)
        raise RuntimeError(f"IK failed for {target}: {np.linalg.norm(err[:3]):.4f}m")

    def _move(self, target_q, gripper):
        start = np.array([self.data.actuator(j.removesuffix("_joint")).ctrl[0] for j in scene.ARM_JOINTS])
        for q in np.linspace(start, target_q, max(12, int(np.max(abs(target_q - start)) / 0.025))):
            for joint, value in zip(scene.ARM_JOINTS, q):
                self.data.actuator(joint.removesuffix("_joint")).ctrl[0] = value
            for _ in range(4):
                self._step(gripper)
        for _ in range(int(0.30 / self.model.opt.timestep)):
            self._step(gripper)

    def _settle(self, gripper, seconds=0.55):
        for _ in range(int(seconds / self.model.opt.timestep)):
            self._step(gripper)

    def run_skill(self, skill, params=None):
        if skill not in SKILLS:
            raise ValueError(f"unknown skill {skill!r}; expected {SKILLS}")
        params, before = dict(params or {}), self.state()
        block_xy = np.array(before["block_pos"][:2])
        target_xy = np.array(params.get("target_xy", before["target_pos"]), dtype=float)
        q = np.array([self.data.joint(j).qpos[0] for j in scene.ARM_JOINTS])
        trace = []
        for phase, point, grip in (
            ("approach", np.r_[block_xy, HOVER_Z], GRIPPER_OPEN),
            ("descend", np.r_[block_xy, GRASP_Z], GRIPPER_OPEN),
        ):
            q = self._ik(point, q); self._move(q, grip); trace.append({"phase": phase, "target": point.tolist()})
        self._settle(GRIPPER_CLOSED); trace.append({"phase": "close", "value": GRIPPER_CLOSED})
        for phase, point in (("lift", np.r_[block_xy, HOVER_Z]), ("move", np.r_[target_xy, TRANSIT_Z]), ("place", np.r_[target_xy, GRASP_Z])):
            q = self._ik(point, q); self._move(q, GRIPPER_CLOSED); trace.append({"phase": phase, "target": point.tolist()})
        self._settle(GRIPPER_OPEN, 0.35); trace.append({"phase": "open", "value": GRIPPER_OPEN})
        q = self._ik(np.r_[target_xy, HOVER_Z], q); self._move(q, GRIPPER_OPEN); trace.append({"phase": "retreat", "target": np.r_[target_xy, HOVER_Z].tolist()})
        after = self.state()
        success = self._max_lift > 0.09 and after["target_distance"] <= TARGET_RADIUS
        failure = None if success else ("missed" if self._max_lift <= 0.09 else "target_miss")
        return Episode(skill, params, trace, before, after, success, failure, np.stack(self._frames), list(self._frame_times))
