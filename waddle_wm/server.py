"""Browser front end for the Claude planner + world-model verifier loop.

    uv run python -m waddle_wm.server

Serves one page: the rendered episode on the left, a chat bar on the right. Type a
command in English, and the page streams back Claude's proposed trace, what the
world model imagined would happen, any repair, and the episode that ran.

Everything runs in this process: one simulator, one verifier, one planner, all owned
by a single worker thread because MuJoCo and the encoder are not re-entrant.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from waddle_wm import chaos
from waddle_wm.agent import DEFAULT_CHECKPOINT, SkillAgent
from waddle_wm.perception import landing_pad
from waddle_wm.planner import MODEL, PlanError, describe_observation

UI = Path(__file__).parent / "ui" / "index.html"
RUNS = Path("results/agent")
DISPLAY = (720, 720)
# UI-only: same 3/4 angle as `demo`, pulled back so the full table is in frame.
DISPLAY_CAMERA = dict(lookat=(0.10, 0.00, 0.00), distance=3.4, azimuth=128, elevation=-28)
DEFAULT_INSTRUCTION = "pick up the red block and put it on the green pad"
MAX_DRAWS = 5          # resample the flaw this many times before falling back to the worst case


def display_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = DISPLAY_CAMERA["lookat"]
    cam.distance = DISPLAY_CAMERA["distance"]
    cam.azimuth = DISPLAY_CAMERA["azimuth"]
    cam.elevation = DISPLAY_CAMERA["elevation"]
    return cam


class Session:
    """One simulator + verifier + planner, owned by a single worker thread.

    MuJoCo's offscreen renderer is bound to the thread that created it, so every
    simulator and encoder call is marshalled onto one worker while the HTTP server
    stays threaded — otherwise a browser's idle keep-alive connection would block
    the next request.
    """

    def __init__(self, checkpoint: Path, seed: int, model: str, repairs: int, threshold: float | None,
                 verifier_mode: str, persistence: int | None = None):
        self.seed, self.refused = seed, None
        self.chaos_saved, self.last_draw, self.pre_chaos_detections = None, None, {}
        self._jobs: queue.Queue = queue.Queue()
        self._display_frames: list = []
        threading.Thread(target=self._worker, daemon=True).start()
        RUNS.mkdir(parents=True, exist_ok=True)
        self.call(lambda: self._build(checkpoint, seed, model, repairs, threshold, verifier_mode,
                                      persistence))

    def _build(self, checkpoint, seed, model, repairs, threshold, verifier_mode, persistence):
        self.agent = SkillAgent(checkpoint, seed, model, repairs, threshold,
                                verifier_mode=verifier_mode, persistence=persistence)
        self.display = mujoco.Renderer(self.agent.env.model, *DISPLAY)
        self.agent.env.on_frame = self._capture_display
        if self.agent.verifier is not None:
            self.agent.observe()          # warm the frozen encoder before the first command
        self.reset(seed)

    def _capture_display(self):
        """Shadow every simulated frame at display resolution.

        The episode the page plays back is rendered at `DISPLAY` rather than the 256 px
        the verifier and the dataset need, which is the only reason this hook exists.
        """
        self.display.update_scene(self.agent.env.data, camera=display_camera())
        self._display_frames.append(self.display.render().copy())

    def _worker(self):
        while True:
            self._jobs.get()()

    def call(self, job):
        """Run `job` on the worker thread and return its value here."""
        box: queue.Queue = queue.Queue()

        def wrapped():
            try:
                box.put(("ok", job()))
            except Exception as error:                  # noqa: BLE001 - reported to the caller
                box.put(("error", error))

        self._jobs.put(wrapped)
        kind, value = box.get()
        if kind == "error":
            raise value
        return value

    def stream(self, instruction: str, *, compare: bool = False, experiment: bool = False):
        """Queue one instruction and yield `(event, payload)` as the worker produces them."""
        events: queue.Queue = queue.Queue()

        def job():
            try:
                if experiment:
                    self.experiment(lambda name, payload: events.put((name, payload)),
                                    str(instruction or DEFAULT_INSTRUCTION))
                elif compare:
                    self.compare(instruction, lambda name, payload: events.put((name, payload)))
                else:
                    self.command(instruction, lambda name, payload: events.put((name, payload)))
            except Exception as error:                  # noqa: BLE001 - a crash must still reach the page
                events.put(("error", {"reason": f"{type(error).__name__}: {error}"}))
                events.put(("done", {"decision": "error", "reason": str(error)}))
            finally:
                events.put(None)

        self._jobs.put(job)
        while (item := events.get()) is not None:
            yield item

    def reset(self, seed: int | None = None):
        self.seed = int(seed) if seed is not None else self.seed + 1
        env = self.agent.env
        env.rng = np.random.default_rng(self.seed)
        self.block_xy = env.sample_scene()      # every command re-runs this scene until it is reset
        self.refused = None
        env.reset(self.block_xy)
        return self.snapshot()

    def snapshot(self) -> dict:
        env = self.agent.env
        detections = self.agent.perceive()
        return {"seed": self.seed,
                "verifier": self.agent.verifier_mode,
                "model": self.agent.model,
                "video": self.preview(),
                "observation": describe_observation(detections, landing_pad(env.model), env.state()["gripper_pos"]),
                "detections": [detection.summary() for detection in detections]}

    def preview(self) -> str:
        """Render a short loop of the current scene for the viewport."""
        frames = []
        for _ in range(16):
            self.display.update_scene(self.agent.env.data, camera=display_camera())
            frames.append(self.display.render().copy())
        video = RUNS / f"preview-{self.seed}-{time.strftime('%H%M%S')}.mp4"
        iio.imwrite(video, np.asarray(frames), fps=4, codec="libx264")
        return f"/runs/{video.name}"

    def _write_video(self, stamp: str, tag: str, run) -> str | None:
        if run.frames is None:
            return None
        # Display capture includes earlier retries; the clip we keep is the last attempt.
        n = len(run.frames)
        frames = self._display_frames[-n:] if len(self._display_frames) >= n else run.frames
        video = RUNS / f"{stamp}-{tag}.mp4"
        iio.imwrite(video, np.asarray(frames), fps=10, codec="libx264")
        return f"/runs/{video.name}"

    def set_verifier(self, mode: str) -> dict:
        """Switch verifier mode on the live agent, same scene, next command.

        The world-model verifier is loaded once at startup regardless of mode, so
        flipping to "none" and back costs nothing — no reload, no re-encoding.
        """
        if mode not in ("none", "rules", "world-model"):
            raise ValueError(f"unknown verifier mode {mode!r}")
        self.agent.verifier_mode = mode
        return {"verifier": mode}

    def set_model(self, model: str) -> dict:
        """Switch the planner model on the live agent. Costs nothing to flip — it's just
        the string handed to `claude -p --model ...` on the next call."""
        if not model:
            raise ValueError("model must not be empty")
        self.agent.model = model
        self.agent.planner.model = model
        return {"model": model}

    def command(self, instruction: str, emit):
        self._run_once(instruction, emit, opening_plan=None)

    def pin_chaos_scene(self, draw, block_xy) -> None:
        """Apply the sampled scene challenge and hold it across every `env.reset` in this run."""
        self.chaos_saved = chaos.apply_chaos_scene(self.agent.env, draw, block_xy)

    def unpin_chaos_scene(self) -> None:
        chaos.restore_chaos_scene(self.agent.env, getattr(self, "chaos_saved", None))
        self.chaos_saved = None

    def _detections(self) -> dict:
        return {detection.label: detection.point_base for detection in self.agent.perceive()}

    def _observation(self) -> str:
        env = self.agent.env
        return describe_observation(self.agent.perceive(), landing_pad(env.model),
                                    env.state()["gripper_pos"])

    def _draw_opening(self, instruction: str, draw, proposed_steps):
        """Pin the sampled scene, then build the flawed opening plan every arm will start from."""
        self.pin_chaos_scene(draw, self.block_xy)
        observed = self._detections()
        # `stale_grasp` is exactly the case where the plan has to be built from what the camera saw
        # *before* the block moved; every other kind plans from the scene as it now stands, so an
        # occluder or a biased camera reaches the plan the way a real sensor error would.
        source = self.pre_chaos_detections if draw.id == "stale_grasp" else observed
        pad_xy, _ = landing_pad(self.agent.env.model)
        if proposed_steps is not None:
            return chaos.build_opening_plan(instruction, source, pad_xy, draw, steps=proposed_steps)
        return chaos.build_opening_plan(instruction, source, pad_xy, draw)

    def _run_baseline(self, instruction: str, opening, stamp: str):
        """The unchecked arm: run the flawed plan with no verifier and no retry, and report it."""
        saved_mode = self.agent.verifier_mode
        self.agent.verifier_mode = "none"
        self._display_frames = []
        try:
            run = self.agent.run(instruction, self.seed, block_xy=self.block_xy, opening_plan=opening,
                                 retry_failed_execution=False, on_event=lambda *_: None)
        finally:
            self.agent.verifier_mode = saved_mode
        return run, self._write_video(stamp, "baseline", run)

    @staticmethod
    def _failed(run) -> bool:
        """Did the unchecked arm actually miss? Anything but a scored success counts as failure."""
        if run.decision != "executed":
            return True
        return (run.execution or {}).get("success") is not True

    def _confirm_run(self, instruction: str, plan, stamp: str):
        """Re-run the plan the loop ended on, from a fresh scene at the same seed.

        The verify/repair loop may repair *from a real miss*, and it deliberately does not reset
        between attempts — the block has moved, and pretending otherwise would be dishonest. That
        makes the last attempt's video start mid-scene, with the block already shoved, which is a
        poor answer to "did the repaired plan actually solve the task?".

        So the demo answers it directly: same seed, same block position, same pinned scene
        challenge, nothing carried over from the failed run, and the repaired plan executed once.
        The verifier still scores it — this is a real run, not a replay of a recording.
        """
        self._display_frames = []
        run = self.agent.run(instruction, self.seed, block_xy=self.block_xy, opening_plan=plan,
                             retry_failed_execution=False, on_event=lambda *_: None)
        return run, self._write_video(stamp, "confirm", run)

    def experiment(self, emit, instruction: str = DEFAULT_INSTRUCTION):
        """The packaged demo: sample a flaw, prove it fails unchecked, then verify and repair it.

        The flaw is drawn fresh on every run (`waddle_wm.chaos`), so two runs back to back do not
        show the same offset twice. A draw is only kept once the unchecked baseline has actually
        missed in MuJoCo — the demo claims the world model prevented a failure, so the failure has
        to be measured rather than asserted. Up to `MAX_DRAWS` draws are tried before falling back
        to the worst case in the band.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S")
        instruction = (instruction or "").strip() or DEFAULT_INSTRUCTION
        saved_mode = self.agent.verifier_mode
        rng = np.random.default_rng()
        compound = chaos.is_compound(instruction)
        proposed_steps, draw, opening, baseline, baseline_video = None, None, None, None, None

        try:
            for attempt in range(MAX_DRAWS):
                emit("phase", {"phase": "sample", "attempt": attempt,
                               "label": "Sampling a flaw..." if attempt == 0
                                        else f"That flaw survived — resampling ({attempt + 1})..."})
                self.unpin_chaos_scene()
                self.agent.env.reset(self.block_xy)
                self.pre_chaos_detections = self._detections()
                seed = int(rng.integers(0, 2 ** 31 - 1))
                draw = chaos.sample(np.random.default_rng(seed), self.pre_chaos_detections,
                                    instruction, seed=seed)
                if not chaos.guarantee_fail(draw) or attempt == MAX_DRAWS - 1:
                    # A draw inside the tolerances would make the demo a coin flip, and by the last
                    # attempt every sampled draw has already survived a real rollout.
                    draw = chaos.worst_case(instruction, seed)
                if compound and proposed_steps is None:
                    # One propose call for the whole experiment: Claude writes the compound plan,
                    # the harness warps exactly one of its steps.
                    emit("phase", {"phase": "propose", "label": "Asking Claude for a multi-step plan..."})
                    proposed_steps = self.agent.planner.propose(instruction, self._observation()).steps
                opening, draw = self._draw_opening(instruction, draw, proposed_steps)
                emit("phase", {"phase": "baseline", "label": "Running the flawed plan unchecked..."})
                baseline, baseline_video = self._run_baseline(instruction, opening, stamp)
                if self._failed(baseline):
                    break

            self.last_draw = draw
            emit("scenario", {"instruction": instruction, "compound": compound,
                              "attempts": attempt + 1, "baseline_failed": self._failed(baseline),
                              **draw.summary()})
            emit("plan", {"kind": "given", "shared": True, "flawed": True,
                          **opening.summary(), "note": opening.note})
            baseline_exec = baseline.execution or {}
            emit("baseline", {"video": baseline_video, "decision": baseline.decision,
                              "execution": baseline_exec,
                              "success": baseline_exec.get("success"),
                              "failure_mode": baseline_exec.get("failure_mode"),
                              "target_distance": baseline_exec.get("target_distance"),
                              "max_block_z": baseline_exec.get("max_block_z")})

            emit("phase", {"phase": "world-model", "label": "Imagining the same plan..."})
            self.agent.verifier_mode = "world-model"
            self._display_frames = []
            wm = self.agent.run(
                instruction, self.seed, block_xy=self.block_xy, opening_plan=opening,
                retry_failed_execution=True,
                on_event=lambda name, payload: emit(name, payload))
            video = self._write_video(stamp, "world-model", wm)
            wm.video = video
            cost_usd = round(sum(call.get("cost_usd") or 0.0 for call in wm.planner_calls), 4)
            wm_exec = wm.execution or {}

            # The clip the page shows for the verified arm is this one: the repaired plan run once
            # from a fresh scene, so success is a solved task rather than a rescued one.
            confirm, confirm_video = None, None
            if wm.decision == "executed" and wm.final_plan is not None and wm.final_plan.executable:
                emit("phase", {"phase": "confirm",
                               "label": "Re-running the repaired plan on a fresh scene..."})
                confirm, confirm_video = self._confirm_run(instruction, wm.final_plan, stamp)
                confirm_exec = confirm.execution or {}
                emit("confirm", {"video": confirm_video, "decision": confirm.decision,
                                 "execution": confirm_exec, "reason": confirm.reason,
                                 "seed": self.seed, "block_xy": [round(v, 4) for v in self.block_xy],
                                 "success": confirm_exec.get("success"),
                                 "failure_mode": confirm_exec.get("failure_mode"),
                                 "target_distance": confirm_exec.get("target_distance"),
                                 "max_block_z": confirm_exec.get("max_block_z")})
                cost_usd = round(cost_usd + sum(call.get("cost_usd") or 0.0
                                                for call in confirm.planner_calls), 4)

            shown = confirm if confirm is not None else wm
            emit("arm-done", {"video": confirm_video or video, "decision": shown.decision,
                              "execution": shown.execution or {}, "fresh_scene": confirm is not None,
                              "reason": shown.reason, "model": wm.model, "cost_usd": cost_usd})
            chaos_record = {"chaos": draw.summary(), "instruction": instruction, "compound": compound}
            (RUNS / f"{stamp}-baseline.json").write_text(json.dumps(
                {**chaos_record, **baseline.as_json()}, indent=2, default=float))
            (RUNS / f"{stamp}-world-model.json").write_text(json.dumps(
                {**chaos_record, **wm.as_json()}, indent=2, default=float))
            if confirm is not None:
                (RUNS / f"{stamp}-confirm.json").write_text(json.dumps(
                    {**chaos_record, "fresh_scene": True, **confirm.as_json()}, indent=2, default=float))
            emit("done", {"decision": "experiment",
                          "baseline_success": baseline_exec.get("success"),
                          "world_model_success": (shown.execution or {}).get("success") is not False
                                                 and shown.decision == "executed",
                          "video": confirm_video or video})
        finally:
            self.agent.verifier_mode = saved_mode
            self.unpin_chaos_scene()
            self.agent.env.reset(self.block_xy)
        self.refused = None

    def compare(self, instruction: str, emit):
        """One Claude plan, two arms: unchecked baseline vs world-model verification."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        instruction = (instruction or "").strip()
        saved_mode = self.agent.verifier_mode
        self._display_frames = []
        self.agent.observe(self.block_xy)
        pad_xy, radius = landing_pad(self.agent.env.model)
        detections = self.agent.perceive()
        observation = describe_observation(
            detections, (pad_xy, radius), self.agent.env.state()["gripper_pos"])
        emit("observed", {"observation": observation, "arm": "shared",
                          "detections": [detection.summary() for detection in detections]})
        try:
            plan = self.agent.planner.propose(instruction, observation)
        except (PlanError, RuntimeError) as error:
            emit("error", {"reason": str(error), "arm": "shared"})
            emit("done", {"decision": "error", "reason": str(error)})
            return
        emit("plan", {"kind": "propose", "shared": True, **plan.summary()})

        videos, execute_budget = {}, None
        try:
            for mode in ("none", "world-model"):
                emit("arm", {"arm": mode})
                self.agent.verifier_mode = mode
                self._display_frames = []
                kwargs = dict(block_xy=self.block_xy, opening_plan=plan, retry_failed_execution=True,
                              on_event=lambda name, payload, arm=mode: emit(name, {**payload, "arm": arm}))
                if mode == "none" and execute_budget is not None:
                    kwargs["execute_budget"] = execute_budget
                run = self.agent.run(instruction, self.seed, **kwargs)
                if mode == "world-model":
                    attempts = (run.execution or {}).get("attempts")
                    execute_budget = len(attempts) if attempts else 1
                video = self._write_video(stamp, mode, run)
                run.video = video
                cost_usd = round(sum(call.get("cost_usd") or 0.0 for call in run.planner_calls), 4)
                videos[mode] = video
                emit("arm-done", {"arm": mode, "video": video, "decision": run.decision,
                                  "execution": run.execution, "reason": run.reason,
                                  "model": run.model, "cost_usd": cost_usd})
                log = RUNS / f"{stamp}-{mode}.json"
                log.write_text(json.dumps(run.as_json(), indent=2, default=float))
        finally:
            self.agent.verifier_mode = saved_mode

        self.refused = None
        emit("done", {"decision": "compared", "videos": videos})

    def _run_once(self, instruction: str, emit, opening_plan):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        instruction = (instruction or "").strip()
        mode = self.agent.verifier_mode
        self._display_frames = []
        kwargs = dict(block_xy=self.block_xy, retry_failed_execution=True, on_event=emit)
        if opening_plan is not None:
            kwargs["opening_plan"] = opening_plan
        run = self.agent.run(instruction, self.seed, **kwargs)
        video = self._write_video(stamp, mode, run)
        run.video = video
        cost_usd = round(sum(call.get("cost_usd") or 0.0 for call in run.planner_calls), 4)
        log = RUNS / f"{stamp}-{mode}.json"
        log.write_text(json.dumps(run.as_json(), indent=2, default=float))

        self.refused = None
        emit("arm-done", {"arm": mode, "video": video, "decision": run.decision, "execution": run.execution,
                          "reason": run.reason, "model": run.model, "cost_usd": cost_usd})
        emit("done", {"decision": run.decision, "video": video, "execution": run.execution, "cost_usd": cost_usd})

    def run_anyway(self) -> dict:
        """Execute the plan the verifier refused, so its prediction can be checked."""
        if self.refused is None:
            return {"error": "no refused plan to run"}
        self._display_frames = []
        execution, frames = self.agent.replay(self.refused, self.block_xy)
        video = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-anyway.mp4"
        display = self._display_frames if len(self._display_frames) == len(frames) else frames
        iio.imwrite(video, np.asarray(display), fps=10, codec="libx264")
        return {"execution": execution, "video": f"/runs/{video.name}"}


class Handler(BaseHTTPRequestHandler):
    session: Session = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"  {self.command} {self.path} -> {fmt % args}", flush=True)

    def _send(self, code, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if content_type.startswith("text/html"):
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]          # the page cache-busts the stream with a query
        if path in ("/", "/index.html"):
            return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/scene":
            snapshot = self.session.call(self.session.snapshot)
            return self._send(200, json.dumps(snapshot).encode(), "application/json")
        if path.startswith("/runs/"):
            video = RUNS / Path(path).name
            if video.suffix == ".mp4" and video.is_file():
                return self._send(200, video.read_bytes(), "video/mp4")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/reset":
            snapshot = self.session.call(lambda: self.session.reset(body.get("seed")))
            return self._send(200, json.dumps(snapshot).encode(), "application/json")
        if self.path == "/api/run-anyway":
            result = self.session.call(self.session.run_anyway)
            return self._send(200, json.dumps(result, default=float).encode(), "application/json")
        if self.path == "/api/verifier":
            try:
                result = self.session.call(lambda: self.session.set_verifier(str(body.get("mode", ""))))
            except ValueError as error:
                return self._send(400, str(error).encode(), "text/plain")
            return self._send(200, json.dumps(result).encode(), "application/json")
        if self.path == "/api/model":
            try:
                result = self.session.call(lambda: self.session.set_model(str(body.get("model", ""))))
            except ValueError as error:
                return self._send(400, str(error).encode(), "text/plain")
            return self._send(200, json.dumps(result).encode(), "application/json")
        if self.path not in ("/api/command", "/api/compare", "/api/experiment"):
            return self._send(404, b"not found", "text/plain")

        compare = self.path == "/api/compare" or body.get("compare") is True
        experiment = self.path == "/api/experiment" or body.get("experiment") is True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        try:
            instruction = str(body.get("instruction", "")).strip()
            for name, payload in self.session.stream(instruction, compare=compare, experiment=experiment):
                chunk = f"data: {json.dumps({'event': name, **payload}, default=float)}\n\n".encode()
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return  # browser refreshed or closed the tab mid-stream


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repairs", type=int, default=2)
    ap.add_argument("--persistence", type=int, default=6,
                    help="how many plans the loop will try per step before giving up. The arm still "
                         "never moves on a rejected plan; this only decides how long the planner "
                         "keeps being asked for a better one")
    ap.add_argument("--threshold", type=float, help="override the checkpoint's calibrated threshold")
    ap.add_argument("--verifier", choices=("none", "rules", "world-model"), default="world-model")
    ap.add_argument("--no-verify", action="store_true", help="deprecated alias for --verifier none")
    args = ap.parse_args()

    mode = "none" if args.no_verify else args.verifier
    print(f"loading the simulator and {mode} verifier ...", flush=True)
    Handler.session = Session(args.checkpoint, args.seed, args.model, args.repairs, args.threshold,
                              mode, persistence=args.persistence)
    print(f"ready: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
