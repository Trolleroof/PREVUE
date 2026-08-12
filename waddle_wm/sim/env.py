"""UR5e/Robotiq skill environment used by the dataset generator."""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from waddle_wm.actions import PHASE_ID
from waddle_wm.sim import relling_scene as scene

SKILLS = ("pick_place",)
APPROACH_DOWN = np.array([0.0, 0.0, -1.0])
HOVER_Z, GRASP_Z, TRANSIT_Z = 0.24, 0.015, 0.30
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 255.0
TARGET_RADIUS = 0.105
LIFT_THRESHOLD = 0.09
FRAMES_TOTAL, PRELUDE_FRAMES, WINDOW_FRAMES = 48, 8, 8
TRACK_KEYS = ("phase", "waypoint", "gripper", "pinch_pos", "block_pos", "all_block_pos",
              "max_block_z", "target_distance")


def pick_place_trace(block_xy, target_xy, grasp_offset_xy=(0.0, 0.0)) -> list[dict]:
    """The waypoint program for `pick_place`, independent of any simulator state.

    The same list is executed by `TabletopEnv.run_skill` and compiled into actions
    by `waddle_wm.actions.compile_plan`, so a candidate plan can be scored without
    touching MuJoCo.
    """
    grasp = np.asarray(block_xy, dtype=float)[:2] + np.asarray(grasp_offset_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)[:2]
    return [{"phase": "approach", "target": [*grasp, HOVER_Z]},
            {"phase": "descend", "target": [*grasp, GRASP_Z]},
            {"phase": "close", "value": GRIPPER_CLOSED},
            {"phase": "lift", "target": [*grasp, HOVER_Z]},
            {"phase": "move", "target": [*target, TRANSIT_Z]},
            {"phase": "place", "target": [*target, GRASP_Z]},
            {"phase": "open", "value": GRIPPER_OPEN},
            {"phase": "retreat", "target": [*target, HOVER_Z]}]


OPEN_PHASES = ("approach", "descend", "retreat")


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
    tracks: dict = field(default_factory=dict)


class TabletopEnv:
    """One physical UR5e pick-and-place execution, rendered at a fixed rate."""

    def __init__(self, camera="demo", width=256, height=256, fps=10, seed=None,
                 block_spawn_low=(0.34, -0.22), block_spawn_high=(0.42, -0.14)):
        self.model = scene.make_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.camera, self.fps = camera, fps
        self.rng = np.random.default_rng(seed)
        self.block_spawn_low = np.asarray(block_spawn_low, dtype=float)
        self.block_spawn_high = np.asarray(block_spawn_high, dtype=float)
        self._frame_steps = max(1, round(1 / (fps * self.model.opt.timestep)))
        self._qadr = np.array([self.model.joint(j).qposadr[0] for j in scene.ARM_JOINTS])
        self._dof = np.array([self.model.joint(j).dofadr[0] for j in scene.ARM_JOINTS])
        self._pinch = self.model.site("2f85/pinch").id
        self._frames, self._frame_times = [], []
        self._max_lift = 0.0
        self.control_delay_steps = 0
        self._control_queue = deque()
        self.on_frame = None        # optional hook, called as each frame is captured (live view)
        self.reset()

    def reset(self, block_xy=None, target_xy=None, blocks=None):
        scene.reset(self.model, self.data)
        if block_xy is not None:
            self.data.joint("red_block_free").qpos[:2] = block_xy
        if blocks is not None:
            for name, position in blocks.items():
                self.data.joint(f"{name}_free").qpos[:3] = position
        if target_xy is not None:
            self.model.site_pos[self.model.site("target").id, :2] = target_xy
        mujoco.mj_forward(self.model, self.data)
        self._tracked_block, self._destination = "red_block", "green_pad"
        self._frames, self._frame_times = [], []
        self._tracks = {key: [] for key in TRACK_KEYS}
        self._phase, self._waypoint = PHASE_ID["idle"], self.data.site("2f85/pinch").xpos.copy()
        self._max_lift = float(self.data.joint("red_block_free").qpos[2])
        self._reset_control_queue()
        return self.state()

    def snapshot(self) -> dict:
        """Everything MuJoCo needs to replay this instant, plus the env's own bookkeeping.

        `mjSTATE_INTEGRATION` is the full integration state — time, qpos, qvel, act, the
        warmstart accelerations, plugin state, ctrl, applied forces, equality flags, mocap
        and userdata — so a restore is byte-identical rather than merely similar. `site_pos`
        lives on the model, not the data, and `reset(target_xy=...)` moves it, so it is
        carried too. Nothing here is ever shown to a selector.
        """
        state = np.zeros(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        return {"state": state, "site_pos": self.model.site_pos.copy(),
                "tracked_block": self._tracked_block, "destination": self._destination,
                "max_lift": float(self._max_lift), "digest": self.state_digest()}

    def restore(self, snapshot: dict):
        """Put the simulator back exactly where `snapshot` was taken, recording cleared.

        Unlike `reset`, this does not re-run the keyframe and re-place the blocks: it writes
        the recorded integration state back, so two candidates executed after two restores
        start from the same bytes and any difference between them is the candidate.
        """
        mujoco.mj_setState(self.model, self.data, snapshot["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        self.model.site_pos[:] = snapshot["site_pos"]
        mujoco.mj_forward(self.model, self.data)
        self._tracked_block, self._destination = snapshot["tracked_block"], snapshot["destination"]
        self._max_lift = float(snapshot["max_lift"])
        self._reset_control_queue()
        self._frames, self._frame_times = [], []
        self._tracks = {key: [] for key in TRACK_KEYS}
        self._phase, self._waypoint = PHASE_ID["idle"], self.data.site("2f85/pinch").xpos.copy()
        return self.state()

    def state_digest(self) -> str:
        """A fingerprint of the physical state, so a failed restore is caught rather than assumed."""
        state = np.zeros(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        payload = np.concatenate([state.ravel(), self.model.site_pos.ravel()]).astype(np.float64)
        return hashlib.sha1(payload.tobytes()).hexdigest()[:16]

    def track_task(self, block: str, destination: str):
        """Select the object and destination whose outcome this execution reports."""
        if block not in scene.BLOCK_NAMES:
            raise ValueError(f"unknown block {block!r}")
        if destination != "green_pad" and destination not in scene.BLOCK_NAMES:
            raise ValueError(f"unknown destination {destination!r}")
        if block == destination:
            raise ValueError("a block cannot be placed onto itself")
        self._tracked_block, self._destination = block, destination
        self._max_lift = float(self.data.joint(f"{block}_free").qpos[2])

    @property
    def frame_count(self) -> int:
        """Frames captured since the last reset, restore, or `clear_recording`."""
        return len(self._frames)

    def clear_recording(self):
        """Start a fresh observation/execution clip without resetting the physical scene."""
        self._frames, self._frame_times = [], []
        self._tracks = {key: [] for key in TRACK_KEYS}

    def home_waypoint(self):
        """Commanded pinch position while the arm is idle at its home pose."""
        return self.data.site("2f85/pinch").xpos.tolist()

    def state(self):
        block = self.data.joint(f"{self._tracked_block}_free").qpos
        grip = self.data.site("2f85/pinch").xpos
        target = (self.model.site_pos[self.model.site("target").id] if self._destination == "green_pad"
                  else self.data.joint(f"{self._destination}_free").qpos[:3])
        return {
            "tracked_block": self._tracked_block,
            "destination": self._destination,
            "block_pos": block[:3].tolist(),
            "block_quat": block[3:7].tolist(),
            "gripper_pos": grip.tolist(),
            "target_pos": target[:2].tolist(),
            "target_z": float(target[2]),
            "target_distance": float(np.linalg.norm(block[:2] - target[:2])),
            "max_block_z": self._max_lift,
        }

    def block_positions(self):
        """Every block on the table, so a planner can be asked about more than the red one."""
        return {name: self.data.joint(f"{name}_free").qpos[:3].tolist() for name in scene.BLOCK_NAMES}

    def sample_scene(self):
        return self.rng.uniform(self.block_spawn_low, self.block_spawn_high)

    def sample_blocks(self):
        """Sample three separated tabletop positions for mixed-object training."""
        positions = {}
        for name in scene.BLOCK_NAMES:
            for _ in range(100):
                xy = self.rng.uniform((0.30, -0.28), (0.68, -0.08))
                if all(np.linalg.norm(xy - np.asarray(other)[:2]) >= 0.075 for other in positions.values()):
                    positions[name] = [*xy, scene.BLOCK_HALF]
                    break
            else:
                raise RuntimeError("could not sample separated block positions")
        return positions

    def _capture(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        self._frames.append(self.renderer.render().copy())
        self._frame_times.append(float(self.data.time))
        state = self.state()
        self._tracks["phase"].append(self._phase)
        self._tracks["waypoint"].append(list(self._waypoint))
        self._tracks["gripper"].append(float(self.data.actuator(scene.GRIPPER_ACTUATOR).ctrl[0]) / GRIPPER_CLOSED)
        for key in ("pinch_pos", "block_pos", "max_block_z", "target_distance"):
            self._tracks[key].append(state["gripper_pos"] if key == "pinch_pos" else state[key])
        self._tracks["all_block_pos"].append([self.data.joint(f"{name}_free").qpos[:3].tolist()
                                              for name in scene.BLOCK_NAMES])
        if self.on_frame is not None:
            self.on_frame()

    def _step(self, gripper, frames=True):
        self.data.actuator(scene.GRIPPER_ACTUATOR).ctrl[0] = gripper
        requested = self.data.ctrl.copy()
        if self.control_delay_steps:
            self._control_queue.append(requested)
            self.data.ctrl[:] = self._control_queue.popleft()
        mujoco.mj_step(self.model, self.data)
        if self.control_delay_steps:
            self.data.ctrl[:] = requested
        self._max_lift = max(self._max_lift, float(self.data.joint(f"{self._tracked_block}_free").qpos[2]))
        if frames and round(self.data.time / self.model.opt.timestep) % self._frame_steps == 0:
            self._capture()

    def set_control_delay(self, steps: int):
        self.control_delay_steps = max(0, int(steps))
        self._reset_control_queue()

    def _reset_control_queue(self):
        self._control_queue = deque(self.data.ctrl.copy() for _ in range(self.control_delay_steps))

    def _ik(self, target, q_init, yaw=None):
        """Damped least squares on the pinch site's position and z-axis.

        `yaw` (radians about vertical, or None) additionally pins the wrist's heading. With
        `yaw=None` — every caller before the code-as-policy programs — the rotation about the
        approach axis stays in the null space and the error term is unchanged.
        """
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self._qadr] = q_init
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        heading = None if yaw is None else np.array([np.cos(yaw), np.sin(yaw), 0.0])
        for _ in range(220):
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)
            pos = scratch.site_xpos[self._pinch]
            frame = scratch.site_xmat[self._pinch].reshape(3, 3)
            z_axis = frame[:, 2]
            rotation = np.cross(z_axis, APPROACH_DOWN)
            if heading is not None:
                # The jaws are symmetric, so aim the x-axis at the heading or its opposite,
                # whichever is nearer: a 180 degree flip is the same grasp.
                x_axis = frame[:, 0]
                wanted = heading if x_axis @ heading >= 0 else -heading
                rotation = 0.5 * (rotation + np.cross(x_axis, wanted))
            err = np.r_[np.asarray(target) - pos, rotation]
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

    def _move(self, target_q, gripper, stop=None):
        start = np.array([self.data.actuator(j.removesuffix("_joint")).ctrl[0] for j in scene.ARM_JOINTS])
        for q in np.linspace(start, target_q, max(12, int(np.max(abs(target_q - start)) / 0.025))):
            for joint, value in zip(scene.ARM_JOINTS, q):
                self.data.actuator(joint.removesuffix("_joint")).ctrl[0] = value
            for _ in range(4):
                self._step(gripper)
                if stop is not None and stop(self):
                    return True
        for _ in range(int(0.30 / self.model.opt.timestep)):
            self._step(gripper)
            if stop is not None and stop(self):     # the arm lags its command, so keep watching while it settles
                return True
        return False

    def approach_until(self, waypoints, gripper=GRIPPER_OPEN, stop=None, phase="approach"):
        """Waypoints + a stop criterion -> a trajectory, stopping early when `stop(env)` holds.

        The pick-and-place path does not use this: its phase durations are what
        `manifest.json -> phase_frames` records, and a contact-triggered early stop would
        desynchronise the compiled plan the verifier scores from the episode that runs.
        It is here for skills that servo to a condition rather than to a pose.
        """
        q = np.array([self.data.joint(j).qpos[0] for j in scene.ARM_JOINTS])
        for point in np.atleast_2d(np.asarray(waypoints, dtype=float)):
            self._begin(phase, point)
            q = self._ik(point, q)
            if self._move(q, gripper, stop):
                return True
        return False

    def pinch_below(self, height):
        """A stop criterion: the gripper's pinch point has descended past `height`."""
        return lambda env: float(env.data.site("2f85/pinch").xpos[2]) <= height

    def _settle(self, gripper, seconds=0.55):
        for _ in range(int(seconds / self.model.opt.timestep)):
            self._step(gripper)

    def _begin(self, name, point=None):
        self._phase = PHASE_ID[name]
        if point is not None:
            self._waypoint = np.asarray(point, dtype=float)
        return len(self._frames)

    def _end(self, name, start, trace, point=None, value=None, yaw=None):
        entry = {"phase": name, "frames": [start, len(self._frames) - 1]}
        if point is not None:
            entry["target"] = np.asarray(point, dtype=float).tolist()
        if value is not None:
            entry["value"] = value
        if yaw is not None:
            entry["yaw"] = float(yaw)
        trace.append(entry)

    def _idle(self, until, gripper=GRIPPER_OPEN):
        self._phase = PHASE_ID["idle"]
        while len(self._frames) < until:
            self._step(gripper)

    def run_skill(self, skill, params=None, frames_total=FRAMES_TOTAL, prelude_frames=PRELUDE_FRAMES):
        """Render `prelude_frames` of the untouched scene, execute the skill, pad to `frames_total`."""
        if skill not in SKILLS:
            raise ValueError(f"unknown skill {skill!r}; expected {SKILLS}")
        params = dict(params or {})
        self.track_task("red_block", "green_pad")
        self._idle(prelude_frames)
        before = self.state()
        plan = pick_place_trace(before["block_pos"], params.get("target_xy", before["target_pos"]),
                                params.get("grasp_offset_xy", (0.0, 0.0)))
        return self._execute(plan, before, skill, params, frames_total)

    def run_trace(self, trace, frames_total=None, prelude_frames=PRELUDE_FRAMES, skill="trace", params=None,
                  block="red_block", destination="green_pad"):
        """Execute an arbitrary waypoint program, the same one `compile_plan` would compile.

        `frames_total=None` lets a free-form trace run as long as it needs; passing
        `FRAMES_TOTAL` keeps the episode on the dataset's frame grid so the verifier's
        compiled rollout and the executed episode describe the same 48 frames.
        """
        return self.run_trace_segments((list(trace),), frames_total, prelude_frames, skill, params,
                                       block, destination)

    def run_trace_segments(self, segments, frames_total=None, prelude_frames=PRELUDE_FRAMES,
                           skill="trace", params=None, block="red_block", destination="green_pad"):
        """Execute lazily produced trace segments as one episode.

        A policy generator may observe the live scene between yielded segments; controller
        state, lift history, recording, and the final outcome remain one atomic attempt.
        """
        self.track_task(block, destination)
        self._idle(prelude_frames)
        return self._execute_segments(segments, self.state(), skill, dict(params or {}), frames_total)

    def _execute(self, plan, before, skill, params, frames_total):
        return self._execute_segments((plan,), before, skill, params, frames_total)

    def _execute_segments(self, segments, before, skill, params, frames_total):
        q = np.array([self.data.joint(j).qpos[0] for j in scene.ARM_JOINTS])
        trace = []
        for plan in segments:
            for entry in plan:
                phase, point = entry["phase"], entry.get("target")
                start = self._begin(phase, point)
                if phase == "close":
                    self._settle(GRIPPER_CLOSED)
                elif phase == "open":
                    self._settle(GRIPPER_OPEN, 0.35)
                elif phase == "idle":
                    self._settle(self.data.actuator(scene.GRIPPER_ACTUATOR).ctrl[0], 0.35)
                else:
                    q = self._ik(point, q, entry.get("yaw"))
                    self._move(q, GRIPPER_OPEN if phase in OPEN_PHASES else GRIPPER_CLOSED)
                self._end(phase, start, trace, point=point, value=entry.get("value"), yaw=entry.get("yaw"))
        if frames_total is not None:
            if len(self._frames) > frames_total:
                raise RuntimeError(f"execution needed {len(self._frames)} frames, over the {frames_total}-frame grid")
            self._idle(frames_total)
        after = self.state()
        radius = TARGET_RADIUS if self._destination == "green_pad" else scene.BLOCK_HALF * 1.5
        stacked = (self._destination == "green_pad" or
                   after["block_pos"][2] >= after["target_z"] + scene.BLOCK_HALF * 1.5)
        success = self._max_lift > LIFT_THRESHOLD and after["target_distance"] <= radius and stacked
        failure = None if success else ("missed" if self._max_lift <= LIFT_THRESHOLD else "target_miss")
        return Episode(skill, params, trace, before, after, success, failure, np.stack(self._frames),
                       list(self._frame_times), {key: list(value) for key, value in self._tracks.items()})

    def observation_frames(self, count=WINDOW_FRAMES):
        """Render `count` frames of the untouched scene: the verifier's observation window."""
        self._idle(count)
        return np.stack(self._frames[:count])
