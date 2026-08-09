"""MuJoCo tabletop environment for Waddle skill-level world models.

One scene (table, block, bin wall, target region, parallel-jaw gripper) driven by
high-level skill traces rather than motor torques. Each episode returns rendered
observation frames, the state before/after, and a ground-truth outcome label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ASSET = Path(__file__).resolve().parent.parent / "assets" / "tabletop.xml"

SKILLS = ("top_grasp", "side_grasp", "align")

# jaw width = distance between the two finger-body origins; the inner faces sit
# 0.010 inside that, so the block (0.042 across) needs > 0.062 to clear.
OPEN_WIDTH = 0.070
CLOSED_WIDTH = 0.030
PUSH_WIDTH = 0.030
GRASP_Z = 0.038
LIFT_Z = 0.20
TRAVEL_Z = 0.22
SIDE_Z = 0.038
YAW_TOP = 0.0
YAW_SIDE = math.pi / 2
TARGET_XY = np.array([-0.22, 0.0])
TARGET_RADIUS = 0.045
LIFT_SUCCESS_Z = 0.09
WALL_Y = 0.077  # inner face of the bin wall


@dataclass
class Episode:
    """One executed skill trace with its observations and ground-truth outcome."""

    skill: str
    params: dict
    skill_trace: list[tuple[str, float]]
    state_before: dict
    state_after: dict
    success: bool
    failure_mode: str | None
    frames: np.ndarray  # (T, H, W, 3) uint8
    frame_times: list[float] = field(default_factory=list)


class TabletopEnv:
    """Skill-level MuJoCo tabletop.

    Physics runs at the model timestep; skills are executed as position setpoint
    segments, and camera frames are captured at a fixed frame rate so clips are
    consistent with V-JEPA 2 preprocessing.
    """

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
        self._gid = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("block", "bin_wall", "table", "palm", "finger_left", "finger_right")
        }
        self._gripper_geoms = {self._gid[g] for g in ("palm", "finger_left", "finger_right")}
        self._finger_geoms = {self._gid[g] for g in ("finger_left", "finger_right")}
        self._block_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")

        self._frames: list[np.ndarray] = []
        self._frame_times: list[float] = []
        self._wall_hit = False
        self._max_block_z = 0.0

    # ------------------------------------------------------------------ state

    def reset(self, block_xy: np.ndarray | None = None) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        if block_xy is None:
            block_xy = np.array([0.0, 0.0])
        self.data.qpos[0:2] = block_xy
        self.data.qpos[2] = 0.021
        self.data.qpos[3:7] = [1, 0, 0, 0]
        # gripper starts parked above the table, jaws open, wrist at the top-grasp yaw
        self.data.qpos[7:11] = [float(block_xy[0]), float(block_xy[1]), TRAVEL_Z, YAW_TOP]
        self.data.qpos[11] = -OPEN_WIDTH / 2
        self.data.qpos[12] = OPEN_WIDTH / 2
        self.data.ctrl[:] = [block_xy[0], block_xy[1], TRAVEL_Z, YAW_TOP, OPEN_WIDTH / 2]
        mujoco.mj_forward(self.model, self.data)

        self._frames = []
        self._frame_times = []
        self._wall_hit = False
        self._max_block_z = float(self.data.xpos[self._block_bid][2])
        return self.state()

    def state(self) -> dict:
        """Structured state vector (§1 of the build plan)."""
        block_pos = self.data.qpos[0:3].copy()
        block_quat = self.data.qpos[3:7].copy()
        grip_xyz = self.data.qpos[7:10].copy()
        width = float(self.data.qpos[12] - self.data.qpos[11])
        return {
            "block_pos": block_pos.tolist(),
            "block_quat": block_quat.tolist(),
            "block_tilt_rad": _tilt_from_quat(block_quat),
            "gripper_pos": grip_xyz.tolist(),
            "gripper_yaw": float(self.data.qpos[10]),
            "gripper_width": width,
            "target_pos": TARGET_XY.tolist(),
            "grasped": bool(self._both_fingers_on_block()),
            "wall_clearance_y": float(WALL_Y - block_pos[1]),
            "target_distance": float(np.linalg.norm(block_pos[:2] - TARGET_XY)),
        }

    # -------------------------------------------------------------- execution

    def _step_segment(self, ctrl: list[float], duration: float) -> None:
        n = max(1, round(duration / self.model.opt.timestep))
        self.data.ctrl[:] = ctrl
        for i in range(n):
            mujoco.mj_step(self.model, self.data)
            self._observe_contacts()
            self._max_block_z = max(self._max_block_z, float(self.data.qpos[2]))
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
            if self._gid["bin_wall"] in pair and pair & self._gripper_geoms:
                self._wall_hit = True

    def _both_fingers_on_block(self) -> bool:
        touching = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self._gid["block"] in pair:
                touching |= pair & self._finger_geoms
        return len(touching) == 2

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

    def _top_grasp(self, p: dict) -> list[tuple[str, float]]:
        """Descend straight down onto the block; the wall is never on the path."""
        bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        dx = p.get("offset_x", 0.0)
        dy = p.get("offset_y", 0.0)
        grip = p.get("grip_width", CLOSED_WIDTH)
        x, y = bx + dx, by + dy
        self._step_segment([x, y, TRAVEL_Z, YAW_TOP, OPEN_WIDTH / 2], 0.5)
        self._step_segment([x, y, GRASP_Z, YAW_TOP, OPEN_WIDTH / 2], 0.7)
        self._step_segment([x, y, GRASP_Z, YAW_TOP, grip / 2], 0.9)
        self._step_segment([x, y, LIFT_Z, YAW_TOP, grip / 2], 1.2)
        return [("approach", TRAVEL_Z), ("descend", GRASP_Z), ("close", grip), ("lift", LIFT_Z)]

    def _side_grasp(self, p: dict) -> list[tuple[str, float]]:
        """Yaw the wrist 90 deg and come in laterally along y at block height.

        ``side=+1`` approaches across the bin wall and is expected to collide.
        """
        bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        side = p.get("side", -1.0)
        standoff = p.get("standoff", 0.13)
        grip = p.get("grip_width", CLOSED_WIDTH)
        entry_y = by + side * standoff
        self._step_segment([bx, entry_y, TRAVEL_Z, YAW_SIDE, OPEN_WIDTH / 2], 0.7)
        self._step_segment([bx, entry_y, SIDE_Z, YAW_SIDE, OPEN_WIDTH / 2], 0.6)
        self._step_segment([bx, by, SIDE_Z, YAW_SIDE, OPEN_WIDTH / 2], 0.9)
        self._step_segment([bx, by, SIDE_Z, YAW_SIDE, grip / 2], 0.9)
        self._step_segment([bx, by, LIFT_Z, YAW_SIDE, grip / 2], 1.2)
        return [
            ("approach", float(entry_y)),
            ("descend", SIDE_Z),
            ("move_in", float(by)),
            ("close", grip),
            ("lift", LIFT_Z),
        ]

    def _align(self, p: dict) -> list[tuple[str, float]]:
        """Push the block along -x into the target region with narrowed jaws."""
        bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        push_to = p.get("push_to_x", TARGET_XY[0])
        standoff = p.get("standoff", 0.09)
        entry_x = bx + standoff
        # the fingers stop against the block's +x face, so drive them past it
        stop_x = push_to + 0.031
        self._step_segment([entry_x, by, TRAVEL_Z, YAW_TOP, PUSH_WIDTH / 2], 0.6)
        self._step_segment([entry_x, by, SIDE_Z, YAW_TOP, PUSH_WIDTH / 2], 0.6)
        self._step_segment([stop_x, by, SIDE_Z, YAW_TOP, PUSH_WIDTH / 2], 2.2)
        self._step_segment([stop_x, by, TRAVEL_Z, YAW_TOP, PUSH_WIDTH / 2], 0.6)
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

        # grasp skills
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
        """Block start pose with enough spread to change which skills work."""
        return np.array(
            [self.rng.uniform(-0.05, 0.045), self.rng.uniform(-0.03, 0.03)]
        )


def _tilt_from_quat(q: np.ndarray) -> float:
    """Angle between the block's local +z and world +z."""
    w, x, y, z = q
    zz = 1 - 2 * (x * x + y * y)
    return float(math.acos(max(-1.0, min(1.0, zz))))
