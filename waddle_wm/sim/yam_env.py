"""MuJoCo tabletop environment driven by a real robot arm (I2RT YAM, 6-DoF).

Same scene, skills, and outcome labels as :mod:`waddle_wm.sim.env`, but the
pseudo-gripper (three slide joints) is replaced by the I2RT YAM manipulator from
MuJoCo Menagerie. Skills are still expressed as Cartesian waypoints; a
damped-least-squares IK solver turns each waypoint into a joint-space setpoint
for the arm's position actuators.

Poses in the public API are in *workspace* coordinates, identical to the
pseudo-gripper env (block starts at the origin, bin wall at y = +0.085, target
at x = -0.22). The workspace sits at ``WORKSPACE_ORIGIN`` in world coordinates
so the arm base has somewhere to stand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from waddle_wm.sim.env import (
    CLOSED_WIDTH,
    Episode,
    LIFT_SUCCESS_Z,
    OPEN_WIDTH,
    PUSH_WIDTH,
    SKILLS,
    TARGET_RADIUS,
    TARGET_XY,
    _tilt_from_quat,
)

ASSET = Path(__file__).resolve().parent.parent / "assets" / "tabletop_yam.xml"

# The tabletop workspace in world coordinates; the arm base is at (0, 0, 0).
# 0.50 m puts the whole skill set inside YAM's usable envelope: every waypoint
# solves to well under a millimetre. See sim/README for the measurements.
WORKSPACE_ORIGIN = np.array([0.50, 0.0, 0.0])
WORKSPACE_YAW = 0.0


def _rz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


WORKSPACE_ROT = _rz(WORKSPACE_YAW)

ARM_DOF = 6
BLOCK_QPOS = 8  # 6 arm joints + 2 finger joints come first

# TCP (the centre of the two gripping plates) as an offset from `grasp_site`,
# expressed in the site frame. Measured off the model: the plates are what close
# on the block, and they sit well below and behind the site. See sim/README.
TCP_OFFSET = np.array([0.0305, 0.0, -0.0247])
# the plates run +-40 mm along the approach axis from the TCP
PLATE_HALF_LENGTH = 0.040

# Heights are TCP heights above the table. For a top-down pose the plates hang
# 40 mm below the TCP, so it sits at 45 mm to keep the tips clear of the table
# while the plate faces still cover most of the 42 mm block. A side grasp comes
# in horizontally, so its TCP sits at the block's mid-height.
# Inner face of the bin wall. Further out than the pseudo-gripper scene's 0.077
# to clear YAM's much bulkier gripper; see the scene XML.
WALL_Y = 0.107

TRAVEL_Z = 0.20
GRASP_Z = 0.045
SIDE_Z = 0.021
LIFT_Z = 0.22
# `align` pushes with the fingertips held horizontally, at the block's mid-height
PUSH_Z = 0.021
FINGERTIP_LEAD = 0.0344  # measured: how far the fingertips lead the TCP

YAW_TOP = 0.0
YAW_SIDE = math.pi / 2

# joint-space seed the IK starts from and the arm rests at (the model's "home")
HOME_QPOS = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])

# Jaw gap commands. The plate faces sit ~0.8 mm outside the commanded gap, so a
# firm grip on the 42 mm block needs a good deal of interference: 0.030 (the
# pseudo-gripper's value) only kisses the faces and the block squirts out on the
# lift, while 0.024 holds it.
YAM_CLOSED_WIDTH = 0.024


def _pose_rotation(approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """World rotation of `grasp_site` from an approach and a jaw-closing axis.

    Site +x is the approach axis: the finger plates hang from the wrist along it,
    so it is the direction the gripper closes down on an object. Site +y is the
    jaw closing axis and site +z follows from the two.
    """
    a = np.asarray(approach, dtype=float)
    a /= np.linalg.norm(a)
    c = np.asarray(closing, dtype=float)
    c = c - np.dot(c, a) * a  # keep the jaw axis perpendicular to the approach
    c /= np.linalg.norm(c)
    return np.column_stack([a, c, np.cross(a, c)])


def top_down_rotation(yaw: float) -> np.ndarray:
    """Approach straight down, jaws closing along an axis yawed about world +z."""
    return _pose_rotation([0.0, 0.0, -1.0], [-math.sin(yaw), math.cos(yaw), 0.0])


def push_rotation(direction: np.ndarray, tilt: float = math.pi / 4) -> np.ndarray:
    """Point the closed fingers along `direction`, tilted `tilt` below horizontal.

    YAM's finger plates protrude only ~1 mm past the wrist housing, so it cannot
    push a 42 mm block with the flat of the plates without the wrist fouling the
    block too; and a purely horizontal fingertip push puts the forearm through
    the table. Angling the fingers down splits the difference: the fingertips
    reach the block below its centre of mass, where a push slides it rather than
    tipping it, while the wrist stays well clear.
    """
    d = np.asarray(direction, dtype=float)
    d = np.array([d[0], d[1], 0.0])
    d /= np.linalg.norm(d)
    approach = math.cos(tilt) * d + math.sin(tilt) * np.array([0.0, 0.0, -1.0])
    return _pose_rotation(approach, np.cross([0.0, 0.0, 1.0], d))


def side_rotation(side: float) -> np.ndarray:
    """Approach horizontally along -side*y, jaws closing along world x.

    This is what a side grasp looks like on a real arm: the whole forearm comes
    in level with the block instead of a top-down wrist roll.
    """
    return _pose_rotation([0.0, -side, 0.0], [1.0, 0.0, 0.0])


class YamTabletopEnv:
    """Skill-level tabletop with a 6-DoF YAM arm under Cartesian IK control."""

    def __init__(
        self,
        camera: str = "frontal",
        width: int = 256,
        height: int = 256,
        fps: int = 10,
        seed: int | None = None,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(ASSET))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.camera = camera
        self.fps = fps
        self.rng = np.random.default_rng(seed)

        self._steps_per_frame = max(1, round((1.0 / fps) / self.model.opt.timestep))
        self._ik_data = mujoco.MjData(self.model)  # scratch for the IK solver

        self._site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self._wrist_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        self._block_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self._block_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "block")
        self._wall_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bin_wall")

        # the YAM finger geoms are unnamed, so group them by owning body
        self._left_geoms = self._geoms_of("link_left_finger", "lf_rot", "lf_down")
        self._right_geoms = self._geoms_of("link_right_finger", "rf_rot", "rf_down")
        self._arm_geoms = self._geoms_of(
            "arm", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6",
            "link_left_finger", "lf_rot", "lf_down",
            "link_right_finger", "rf_rot", "rf_down",
        )

        self._frames: list[np.ndarray] = []
        self._frame_times: list[float] = []
        self._wall_hit = False
        self._max_block_z = 0.0
        self._ik_residual = 0.0

    def _geoms_of(self, *body_names: str) -> set[int]:
        bids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in body_names
        }
        return {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in bids}

    # ------------------------------------------------------------------ frames

    def to_world(self, xy_or_xyz) -> np.ndarray:
        v = np.asarray(xy_or_xyz, dtype=float)
        flat = v.shape == (2,)
        v3 = np.array([v[0], v[1], 0.0]) if flat else v
        w = WORKSPACE_ORIGIN + WORKSPACE_ROT @ v3
        return w[:2] if flat else w

    def to_local(self, xy_or_xyz) -> np.ndarray:
        v = np.asarray(xy_or_xyz, dtype=float)
        flat = v.shape == (2,)
        v3 = np.array([v[0], v[1], 0.0]) if flat else v
        loc = WORKSPACE_ROT.T @ (v3 - WORKSPACE_ORIGIN)
        return loc[:2] if flat else loc

    # ------------------------------------------------------------------- state

    def reset(self, block_xy: np.ndarray | None = None) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        if block_xy is None:
            block_xy = np.array([0.0, 0.0])
        world_xy = self.to_world(np.asarray(block_xy, dtype=float))

        self.data.qpos[:ARM_DOF] = HOME_QPOS
        self.data.qpos[6] = OPEN_WIDTH / 2
        self.data.qpos[7] = -OPEN_WIDTH / 2
        self.data.qpos[BLOCK_QPOS + 0 : BLOCK_QPOS + 2] = world_xy
        self.data.qpos[BLOCK_QPOS + 2] = 0.021
        self.data.qpos[BLOCK_QPOS + 3 : BLOCK_QPOS + 7] = [1, 0, 0, 0]
        self.data.ctrl[:ARM_DOF] = HOME_QPOS
        self.data.ctrl[ARM_DOF] = OPEN_WIDTH / 2
        mujoco.mj_forward(self.model, self.data)

        self._frames = []
        self._frame_times = []
        self._wall_hit = False
        self._max_block_z = float(self.data.xpos[self._block_bid][2])
        self._ik_residual = 0.0
        return self.state()

    def tcp(self) -> np.ndarray:
        """World position of the point between the finger pads."""
        r = self.data.site_xmat[self._site].reshape(3, 3)
        return self.data.site_xpos[self._site] + r @ TCP_OFFSET

    def state(self) -> dict:
        block_pos = self.to_local(self.data.qpos[BLOCK_QPOS : BLOCK_QPOS + 3].copy())
        block_quat = self.data.qpos[BLOCK_QPOS + 3 : BLOCK_QPOS + 7].copy()
        tcp = self.to_local(self.tcp())
        r = WORKSPACE_ROT.T @ self.data.site_xmat[self._site].reshape(3, 3)
        yaw = float(math.atan2(r[1, 1], -r[0, 1]))  # jaw closing axis, workspace frame
        width = float(self.data.qpos[6] - self.data.qpos[7])
        return {
            "block_pos": block_pos.tolist(),
            "block_quat": block_quat.tolist(),
            "block_tilt_rad": _tilt_from_quat(block_quat),
            "gripper_pos": tcp.tolist(),
            "gripper_yaw": yaw,
            "gripper_width": width,
            "arm_qpos": self.data.qpos[:ARM_DOF].tolist(),
            "target_pos": TARGET_XY.tolist(),
            "grasped": bool(self._both_fingers_on_block()),
            "wall_clearance_y": float(WALL_Y - block_pos[1]),
            "target_distance": float(np.linalg.norm(block_pos[:2] - TARGET_XY)),
        }

    # ---------------------------------------------------------------------- IK

    def solve_ik(
        self,
        tcp_local: np.ndarray,
        rot: np.ndarray,
        seed: np.ndarray | None = None,
        iters: int = 200,
        damping: float = 0.08,
        restarts: int = 24,
        rot_weight: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Damped-least-squares IK for a TCP position + top-down yaw.

        The arm is redundant enough that a single descent often stalls in a
        local minimum, so this runs several seeded restarts (the caller's seed,
        the home pose, then deterministic random configurations) and keeps the
        best solution. Returns the joint vector and the position error in metres.
        """
        target = self.to_world(np.asarray(tcp_local, dtype=float))
        seeds = [np.array(HOME_QPOS, dtype=float)]
        if seed is not None:
            seeds.insert(0, np.array(seed, dtype=float))
        seeds.extend(self._nearest_seeds(target, np.asarray(rot, dtype=float), restarts))

        solutions = []
        for s in seeds:
            q, err = self._ik_descent(tcp_local, rot, s, iters, damping, rot_weight)
            solutions.append((q, err))
            if err < 1e-4 and seed is not None:
                # a good solution from the continuation seed is the one we want
                if s is seeds[0]:
                    return q, err
            if len(solutions) >= 4 and min(e for _, e in solutions) < 1e-4:
                break

        best_err = min(e for _, e in solutions)
        # Among solutions that reach the target, prefer the one closest to the
        # current arm configuration. Waypoints are ramped in joint space, so an
        # elbow flip between two waypoints sweeps the whole arm through the
        # scene even though both endpoints are fine.
        tol = max(best_err + 1e-3, 2e-3)
        reference = np.array(seed if seed is not None else HOME_QPOS, dtype=float)
        viable = [(q, e) for q, e in solutions if e <= tol]
        best_q, best_err = min(viable, key=lambda qe: np.linalg.norm(qe[0] - reference))
        return best_q, best_err

    def _nearest_seeds(self, target: np.ndarray, rot: np.ndarray, k: int) -> list[np.ndarray]:
        """Seed configurations from the pose library, nearest to the goal first.

        Poses that point the gripper down near the table occupy a thin sliver of
        this arm's configuration space, so random restarts almost never land in
        the right basin; seeding from sampled forward kinematics does.
        """
        tcps, approaches, qs = self._pose_library()
        # metre-scale distance plus an angular term on the approach axis
        dist = np.linalg.norm(tcps - target, axis=1)
        align = 1.0 - approaches @ rot[:, 2]
        order = np.argsort(dist + 0.25 * align)[:k]
        return [qs[i].copy() for i in order]

    @classmethod
    def _pose_library(cls) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sampled (TCP, approach axis, joints) triples, built once per process."""
        cached = getattr(cls, "_POSE_LIB", None)
        if cached is not None:
            return cached

        model = mujoco.MjModel.from_xml_path(str(ASSET))
        data = mujoco.MjData(model)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        lo = model.jnt_range[:ARM_DOF, 0]
        hi = model.jnt_range[:ARM_DOF, 1]
        rng = np.random.default_rng(0)  # fixed, so the library is reproducible
        n = 40000
        samples = rng.uniform(lo, hi, size=(n, ARM_DOF))

        tcps = np.zeros((n, 3))
        approaches = np.zeros((n, 3))
        for i, q in enumerate(samples):
            data.qpos[:ARM_DOF] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            r = data.site_xmat[site].reshape(3, 3)
            tcps[i] = data.site_xpos[site] + r @ TCP_OFFSET
            approaches[i] = r[:, 2]

        keep = (tcps[:, 2] > -0.02) & (tcps[:, 2] < 0.40)
        lib = (tcps[keep], approaches[keep], samples[keep])
        cls._POSE_LIB = lib
        return lib

    @staticmethod
    def _pose_error(
        r: np.ndarray,
        tcp: np.ndarray,
        target: np.ndarray,
        q_des: np.ndarray,
        scratch: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        q_cur, q_err, rot_err = scratch
        mujoco.mju_mat2Quat(q_cur, r.flatten())
        mujoco.mju_negQuat(q_cur, q_cur)
        mujoco.mju_mulQuat(q_err, q_des, q_cur)
        mujoco.mju_quat2Vel(rot_err, q_err, 1.0)
        return target - tcp, rot_err.copy()

    def _ik_descent(
        self,
        tcp_local: np.ndarray,
        rot: np.ndarray,
        seed: np.ndarray,
        iters: int,
        damping: float,
        rot_weight: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        target = self.to_world(np.asarray(tcp_local, dtype=float))
        r_des = np.asarray(rot, dtype=float)
        q = np.array(seed, dtype=float)

        d = self._ik_data
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        q_cur = np.zeros(4)
        q_des = np.zeros(4)
        q_err = np.zeros(4)
        rot_err = np.zeros(3)
        mujoco.mju_mat2Quat(q_des, r_des.flatten())

        lo = self.model.jnt_range[:ARM_DOF, 0]
        hi = self.model.jnt_range[:ARM_DOF, 1]

        scratch = (q_cur, q_err, rot_err)

        def cost_at(qq: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
            d.qpos[:] = self.data.qpos
            d.qpos[:ARM_DOF] = qq
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            r = d.site_xmat[self._site].reshape(3, 3)
            tcp = d.site_xpos[self._site] + r @ TCP_OFFSET
            pe, re = self._pose_error(r, tcp, target, q_des, scratch)
            # position dominates the accept/reject test: orientation is the
            # secondary task and must never be traded for reach
            weighted = np.concatenate([pe, 0.02 * rot_weight * re])
            return float(weighted @ weighted), pe, re, tcp

        cost, pos_err, rot_err_v, tcp = cost_at(q)
        lam = damping
        for _ in range(iters):
            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err_v) < 1e-3:
                break
            mujoco.mj_jac(self.model, d, jacp, jacr, tcp, self._wrist_bid)
            jp = jacp[:, :ARM_DOF]
            jr = jacr[:, :ARM_DOF]

            # Task-priority IK: YAM cannot hold an exactly vertical approach at
            # most tabletop positions, but it can always reach the position, so
            # position is the primary task and orientation is solved only in the
            # position nullspace. A joint on its limit is dropped from both, or
            # it soaks up the step pushing into the limit and the descent stalls.
            active = np.ones(ARM_DOF, dtype=bool)
            for _ in range(ARM_DOF):
                jpa, jra = jp * active, jr * active
                pinv_p = jpa.T @ np.linalg.solve(jpa @ jpa.T + lam**2 * np.eye(3), np.eye(3))
                null = np.eye(ARM_DOF) - pinv_p @ jpa
                dq_r = jra.T @ np.linalg.solve(
                    jra @ jra.T + lam**2 * np.eye(3), rot_weight * rot_err_v
                )
                dq = pinv_p @ pos_err + null @ dq_r
                pinned = ((q <= lo + 1e-9) & (dq < 0)) | ((q >= hi - 1e-9) & (dq > 0))
                if not (pinned & active).any():
                    break
                active &= ~pinned
            dq = np.where(active, dq, 0.0)

            q_try = np.clip(q + np.clip(dq, -0.4, 0.4), lo, hi)
            cost_try, pe_try, re_try, tcp_try = cost_at(q_try)
            if cost_try < cost:
                q, cost, pos_err, rot_err_v, tcp = q_try, cost_try, pe_try, re_try, tcp_try
                lam = max(lam * 0.7, 1e-3)
            else:
                lam *= 2.5
                if lam > 1e4:  # the step is vanishing; this basin is exhausted
                    break

        return q, float(np.linalg.norm(pos_err))

    # -------------------------------------------------------------- execution

    def _move_to(
        self,
        tcp_local,
        rot: np.ndarray,
        grip: float,
        duration: float,
        waypoints: int = 6,
        settle: float = 0.35,
    ) -> np.ndarray:
        """Drive the TCP to a pose along a straight Cartesian line.

        The line is split into sub-waypoints that are solved in sequence, each
        seeded from the last, so the gripper actually travels through the
        workspace instead of taking whatever arc a joint-space ramp produces.
        That matters here: the bin wall only blocks a side grasp if the arm
        really moves through the space the wall occupies.
        """
        start = self.to_local(self.tcp())
        goal = np.asarray(tcp_local, dtype=float)
        grip_start = float(self.data.ctrl[ARM_DOF])
        seed = self.data.ctrl[:ARM_DOF].copy()

        q_goal = seed
        for k in range(1, waypoints + 1):
            frac = k / waypoints
            via = start + frac * (goal - start)
            q_goal, residual = self.solve_ik(via, rot, seed=seed)
            self._ik_residual = max(self._ik_residual, residual)
            self._ramp_to(q_goal, grip_start + frac * (grip / 2 - grip_start),
                          duration / waypoints)
            seed = q_goal

        # Hold the final setpoint briefly. The position servos lag by a centimetre
        # or so while a segment is still ramping, and skills whose contact
        # geometry depends on height (pushing, descending onto the block) need
        # the arm to have actually arrived before the next phase starts.
        self._ramp_to(q_goal, grip / 2, settle)
        return q_goal

    def _ramp_to(self, q_goal: np.ndarray, grip_ctrl: float, duration: float) -> None:
        q_start = self.data.ctrl[:ARM_DOF].copy()
        grip_start = float(self.data.ctrl[ARM_DOF])
        n = max(1, round(duration / self.model.opt.timestep))
        for i in range(1, n + 1):
            a = i / n
            self.data.ctrl[:ARM_DOF] = q_start + a * (q_goal - q_start)
            self.data.ctrl[ARM_DOF] = grip_start + a * (grip_ctrl - grip_start)
            mujoco.mj_step(self.model, self.data)
            self._observe_contacts()
            self._max_block_z = max(
                self._max_block_z, float(self.data.qpos[BLOCK_QPOS + 2])
            )
            if (self._sim_step_count() % self._steps_per_frame) == 0:
                self._capture()

    def _hold(self, grip: float, duration: float) -> None:
        """Keep the arm setpoint, ramp only the jaws (used for closing)."""
        grip_start = float(self.data.ctrl[ARM_DOF])
        n = max(1, round(duration / self.model.opt.timestep))
        for i in range(1, n + 1):
            a = i / n
            self.data.ctrl[ARM_DOF] = grip_start + a * (grip / 2 - grip_start)
            mujoco.mj_step(self.model, self.data)
            self._observe_contacts()
            self._max_block_z = max(
                self._max_block_z, float(self.data.qpos[BLOCK_QPOS + 2])
            )
            if (self._sim_step_count() % self._steps_per_frame) == 0:
                self._capture()

    def _sim_step_count(self) -> int:
        return round(self.data.time / self.model.opt.timestep)

    def _capture(self) -> None:
        self.renderer.update_scene(self.data, camera=self.camera)
        self._frames.append(self.renderer.render().copy())
        self._frame_times.append(float(self.data.time))

    def _observe_contacts(self) -> None:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self._wall_gid in pair and pair & self._arm_geoms:
                self._wall_hit = True

    def _both_fingers_on_block(self) -> bool:
        left = right = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self._block_gid not in pair:
                continue
            other = pair - {self._block_gid}
            left = left or bool(other & self._left_geoms)
            right = right or bool(other & self._right_geoms)
        return left and right

    # ------------------------------------------------------------------ skills

    def run_skill(self, skill: str, params: dict | None = None) -> Episode:
        if skill not in SKILLS:
            raise ValueError(f"unknown skill {skill!r}; expected one of {SKILLS}")
        params = dict(params or {})
        state_before = self.state()
        trace = getattr(self, f"_{skill}")(params)
        state_after = self.state()
        success, failure_mode = self._score(skill, state_before, state_after)
        return Episode(
            skill=skill,
            params=params,
            skill_trace=trace,
            state_before=state_before,
            state_after=state_after,
            success=success,
            failure_mode=failure_mode,
            frames=np.stack(self._frames) if self._frames else np.empty((0, 0, 0, 3), np.uint8),
            frame_times=list(self._frame_times),
        )

    def _block_xy(self) -> tuple[float, float]:
        p = self.to_local(self.data.qpos[BLOCK_QPOS : BLOCK_QPOS + 3].copy())
        return float(p[0]), float(p[1])

    def _top_grasp(self, p: dict) -> list[tuple[str, float]]:
        bx, by = self._block_xy()
        x = bx + p.get("offset_x", 0.0)
        y = by + p.get("offset_y", 0.0)
        grip = p.get("grip_width", YAM_CLOSED_WIDTH)
        self._move_to([x, y, TRAVEL_Z], top_down_rotation(YAW_TOP), OPEN_WIDTH, 1.0)
        self._move_to([x, y, GRASP_Z], top_down_rotation(YAW_TOP), OPEN_WIDTH, 1.0)
        self._hold(grip, 0.8)
        self._move_to([x, y, LIFT_Z], top_down_rotation(YAW_TOP), grip, 1.4)
        return [("approach", TRAVEL_Z), ("descend", GRASP_Z), ("close", grip), ("lift", LIFT_Z)]

    def _side_grasp(self, p: dict) -> list[tuple[str, float]]:
        """Yaw the jaws 90 deg and come in laterally along y at block height.

        ``side=+1`` approaches across the bin wall and is expected to collide.
        """
        bx, by = self._block_xy()
        side = p.get("side", -1.0)
        standoff = p.get("standoff", 0.13)
        grip = p.get("grip_width", YAM_CLOSED_WIDTH)
        entry_y = by + side * standoff
        rot = WORKSPACE_ROT @ side_rotation(side)  # identity yaw today, kept explicit
        self._move_to([bx, entry_y, TRAVEL_Z], rot, OPEN_WIDTH, 1.0)
        self._move_to([bx, entry_y, SIDE_Z], rot, OPEN_WIDTH, 0.9)
        self._move_to([bx, by, SIDE_Z], rot, OPEN_WIDTH, 1.2)
        self._hold(grip, 0.8)
        self._move_to([bx, by, LIFT_Z], rot, grip, 1.4)
        return [
            ("approach", float(entry_y)),
            ("descend", SIDE_Z),
            ("move_in", float(by)),
            ("close", grip),
            ("lift", LIFT_Z),
        ]

    def _align(self, p: dict) -> list[tuple[str, float]]:
        """Push the block along -x into the target region with narrowed jaws."""
        bx, by = self._block_xy()
        push_to = p.get("push_to_x", TARGET_XY[0])
        standoff = p.get("standoff", 0.09)
        entry_x = bx + standoff
        # the fingertips lead the TCP by FINGERTIP_LEAD and stop against the
        # block's +x face, so the TCP stops that much short of the goal
        stop_x = push_to + 0.021 + FINGERTIP_LEAD
        rot = push_rotation([-1.0, 0.0, 0.0])
        self._move_to([entry_x, by, TRAVEL_Z], rot, PUSH_WIDTH, 1.0)
        self._move_to([entry_x, by, PUSH_Z], rot, PUSH_WIDTH, 0.9)
        self._move_to([stop_x, by, PUSH_Z], rot, PUSH_WIDTH, 2.4)
        self._move_to([stop_x, by, TRAVEL_Z], rot, PUSH_WIDTH, 1.0)
        return [("approach", float(entry_x)), ("move", float(push_to)), ("settle", TRAVEL_Z)]

    # ------------------------------------------------------------------ labels

    def _score(self, skill: str, before: dict, after: dict) -> tuple[bool, str | None]:
        if skill == "align":
            if after["block_tilt_rad"] > 0.5:
                return False, "toppled"
            d = after["target_distance"]
            if d <= TARGET_RADIUS:
                return True, None
            moved = np.linalg.norm(
                np.array(after["block_pos"][:2]) - np.array(before["block_pos"][:2])
            )
            if moved < 0.01:
                return False, "no_contact"
            overshot = after["block_pos"][0] < TARGET_XY[0] - TARGET_RADIUS
            return False, "overshoot" if overshot else "undershoot"

        lifted = after["block_pos"][2] > LIFT_SUCCESS_Z
        if lifted and after["grasped"]:
            return True, None
        if self._wall_hit:
            return False, "collision_with_bin_wall"
        if self._max_block_z > 0.06 and not lifted:
            return False, "slip"
        if after["grasped"]:
            return False, "grip_too_weak"
        return False, "missed"

    # ------------------------------------------------------------- randomizers

    def sample_scene(self) -> np.ndarray:
        return np.array([self.rng.uniform(-0.05, 0.045), self.rng.uniform(-0.03, 0.03)])
