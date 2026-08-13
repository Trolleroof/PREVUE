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

from waddle_wm.agent import DEFAULT_CHECKPOINT, SkillAgent
from waddle_wm.perception import landing_pad
from waddle_wm.planner import MODEL, PlanError, describe_observation

UI = Path(__file__).parent / "ui" / "index.html"
RUNS = Path("results/agent")
DISPLAY = (720, 720)
# UI-only: same 3/4 angle as `demo`, pulled back so the full table is in frame.
DISPLAY_CAMERA = dict(lookat=(0.10, 0.00, 0.00), distance=3.4, azimuth=128, elevation=-28)


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

    def __init__(self, checkpoint: Path, seed: int, model: str, repairs: int, threshold: float | None, verifier_mode: str):
        self.seed, self.refused = seed, None
        self._jobs: queue.Queue = queue.Queue()
        self._display_frames: list = []
        threading.Thread(target=self._worker, daemon=True).start()
        RUNS.mkdir(parents=True, exist_ok=True)
        self.call(lambda: self._build(checkpoint, seed, model, repairs, threshold, verifier_mode))

    def _build(self, checkpoint, seed, model, repairs, threshold, verifier_mode):
        self.agent = SkillAgent(checkpoint, seed, model, repairs, threshold, verifier_mode=verifier_mode)
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

    def stream(self, instruction: str):
        """Queue one instruction and yield `(event, payload)` as the worker produces them."""
        events: queue.Queue = queue.Queue()

        def job():
            try:
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

    def command(self, instruction: str, emit):
        """Same user instruction, twice: world model vs unverified."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        saved_mode = self.agent.verifier_mode
        instruction = (instruction or "").strip()
        self.agent.observe(self.block_xy)
        pad_xy, radius = landing_pad(self.agent.env.model)
        observation = describe_observation(
            self.agent.perceive(), (pad_xy, radius), self.agent.env.state()["gripper_pos"])
        emit("observed", {"observation": observation})
        try:
            plan = self.agent.planner.propose(instruction, observation)
        except (PlanError, RuntimeError) as error:
            emit("error", {"reason": str(error)})
            emit("done", {"decision": "error", "reason": str(error), "videos": {}})
            return
        emit("plan", {"kind": "propose", "shared": True, **plan.summary()})

        videos, results = {}, {}
        execute_budget = None
        try:
            # Same Claude plan, two arms: world model imagines and may repair; baseline runs it.
            for mode, tag in (("world-model", "world-model"), ("none", "unverified")):
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
                video = self._write_video(stamp, tag, run)
                run.video = video
                videos[mode] = video
                results[mode] = {"decision": run.decision, "execution": run.execution, "reason": run.reason}
                emit("arm-done", {"arm": mode, "video": video, **results[mode]})
                log = RUNS / f"{stamp}-{tag}.json"
                log.write_text(json.dumps(run.as_json(), indent=2, default=float))
        finally:
            self.agent.verifier_mode = saved_mode

        self.refused = None
        emit("done", {"decision": "compared", "videos": videos, "results": results, "can_run_anyway": False})

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
        if self.path != "/api/command":
            return self._send(404, b"not found", "text/plain")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        try:
            for name, payload in self.session.stream(str(body.get("instruction", "")).strip()):
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
    ap.add_argument("--threshold", type=float, help="override the checkpoint's calibrated threshold")
    ap.add_argument("--verifier", choices=("none", "rules", "world-model"), default="world-model")
    ap.add_argument("--no-verify", action="store_true", help="deprecated alias for --verifier none")
    args = ap.parse_args()

    mode = "none" if args.no_verify else args.verifier
    print(f"loading the simulator and {mode} verifier ...", flush=True)
    Handler.session = Session(args.checkpoint, args.seed, args.model, args.repairs, args.threshold, mode)
    print(f"ready: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
