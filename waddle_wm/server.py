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
from waddle_wm.planner import MODEL, describe_observation

UI = Path(__file__).parent / "ui" / "index.html"
RUNS = Path("results/agent")
DISPLAY = (720, 720)


class Session:
    """One simulator + verifier + planner, owned by a single worker thread.

    MuJoCo's offscreen renderer is bound to the thread that created it, so every
    simulator and encoder call is marshalled onto one worker while the HTTP server
    stays threaded — otherwise a browser's idle keep-alive connection would block
    the next request.
    """

    def __init__(self, checkpoint: Path, seed: int, model: str, repairs: int, threshold: float, verify: bool):
        self.seed, self.refused = seed, None
        self._jobs: queue.Queue = queue.Queue()
        self._display_frames: list = []
        threading.Thread(target=self._worker, daemon=True).start()
        RUNS.mkdir(parents=True, exist_ok=True)
        self.call(lambda: self._build(checkpoint, seed, model, repairs, threshold, verify))

    def _build(self, checkpoint, seed, model, repairs, threshold, verify):
        self.agent = SkillAgent(checkpoint, seed, model, repairs, threshold, verify)
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
        self.display.update_scene(self.agent.env.data, camera=self.agent.env.camera)
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
        if seed is not None:
            self.seed = seed
            self.agent.env.rng = np.random.default_rng(seed)
        env = self.agent.env
        self.block_xy = env.sample_scene()      # every command re-runs this scene until it is reset
        self.refused = None
        env.reset(self.block_xy)
        return self.snapshot()

    def snapshot(self) -> dict:
        env = self.agent.env
        detections = self.agent.perceive()
        return {"seed": self.seed,
                "observation": describe_observation(detections, landing_pad(env.model), env.state()["gripper_pos"]),
                "detections": [detection.summary() for detection in detections]}

    def command(self, instruction: str, emit):
        """Run one instruction end to end, calling `emit(name, payload)` as it goes."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self._display_frames = []
        run = self.agent.run(instruction, self.seed, block_xy=self.block_xy, on_event=emit)
        payload = run.as_json()
        if run.frames is not None:
            frames = self._display_frames if len(self._display_frames) == len(run.frames) else run.frames
            video = RUNS / f"{stamp}.mp4"
            iio.imwrite(video, np.asarray(frames), fps=10, codec="libx264")
            run.video = payload["video"] = f"/runs/{video.name}"
        log = RUNS / f"{stamp}.json"
        log.write_text(json.dumps(payload, indent=2, default=float))
        payload["log"] = str(log)
        # A rejected plan never ran, so the failure the verifier predicted is invisible
        # until someone asks for it. Keep it around so the page can offer exactly that.
        self.refused = run.final_plan if run.decision == "rejected" else None
        payload["can_run_anyway"] = self.refused is not None
        emit("done", payload)

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

        for name, payload in self.session.stream(str(body.get("instruction", "")).strip()):
            chunk = f"data: {json.dumps({'event': name, **payload}, default=float)}\n\n".encode()
            self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repairs", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-verify", action="store_true", help="LLM-only baseline: skip the world model")
    args = ap.parse_args()

    print("loading the simulator, the verifier and the frozen encoder ...", flush=True)
    Handler.session = Session(args.checkpoint, args.seed, args.model, args.repairs, args.threshold, not args.no_verify)
    print(f"ready: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
