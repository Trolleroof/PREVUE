"""Pre-execution skill verifier: score a candidate plan by imagining it.

The verifier encodes one 8-frame observation window of the *untouched* scene,
compiles the proposed skill trace into action chunks, rolls the latent dynamics
ensemble forward over the whole plan, and decodes the imagined terminal latent.
Nothing about the real execution is consulted — no cached outcome, no label.

    uv run python -m waddle_wm.verifier --episode ur5e_0007 --target-x 0.5 --target-y 0.3
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from waddle_wm.actions import ACTION_DIM, chunks, compile_plan
from waddle_wm.embed_windows import clip_frames
from waddle_wm.sim.env import pick_place_trace
from waddle_wm.train_latent_dynamics import STATE_DIM, Dynamics, Readout, rollout, success_probability


def through_codec(frames, fps: int = 10) -> list[np.ndarray]:
    """Round-trip freshly rendered frames through h264, the way the training clips were stored.

    Every cached window embedding was computed from a decoded `.mp4`, so handing the frozen
    backbone raw renderer output puts it off-distribution. It is not a small effect: on three
    scenes whose canonical plan really succeeds, the same plan scored p(success) 0.17 / 0.30 /
    0.12 from raw frames and 0.98 / 0.86 / 1.00 through the codec. Any live camera window has
    to take the same path the dataset took.
    """
    import imageio.v3 as iio
    from waddle_wm.embed_windows import clip_frames

    frames = np.asarray(frames)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "window.mp4"
        iio.imwrite(path, frames, fps=fps, codec="libx264")
        return clip_frames(path, len(frames))


@dataclass
class VerificationResult:
    approve: bool
    success_probability: float
    uncertainty: float
    lifted_probability: float
    in_target_probability: float
    predicted_block_xy: list[float]
    likely_failure: str | None
    suggestion: str | None


class Verifier:
    """`verify(observation_window, skill_trace) -> VerificationResult`."""

    def __init__(self, checkpoint: Path, model: Path = Path("models/vjepa2-vitl-fpc64-256"), threshold: float = 0.5, device=None):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.manifest, self.threshold, self.model_path = saved["manifest"], threshold, model
        self.members = nn.ModuleList([Dynamics(saved["latent_dim"], saved["chunk_dim"]) for _ in range(saved["member_count"])])
        self.members.load_state_dict(saved["members"]); self.members.to(self.device).eval()
        self.readout = Readout(saved["latent_dim"]); self.readout.load_state_dict(saved["readout"]); self.readout.to(self.device).eval()
        self.norm = {key: value.to(self.device) for key, value in saved["normalization"].items()}
        self._encoder = None

    def encode(self, frames) -> torch.Tensor:
        """One observation window of raw RGB frames -> one normalised latent."""
        window = self.manifest["window_frames"]
        if len(frames) != window:
            raise ValueError(f"observation window must be {window} frames, got {len(frames)}")
        if self._encoder is None:
            from transformers import AutoModel, AutoVideoProcessor
            self._encoder = (AutoVideoProcessor.from_pretrained(self.model_path, local_files_only=True),
                             AutoModel.from_pretrained(self.model_path, local_files_only=True).to(self.device).eval())
        processor, encoder = self._encoder
        with torch.inference_mode():
            pixels = processor(list(frames), return_tensors="pt")["pixel_values_videos"].to(self.device)
            latent = encoder(pixel_values_videos=pixels).last_hidden_state.mean(dim=1).float()
        return (latent - self.norm["latent_mean"]) / self.norm["latent_std"]

    def encode_live(self, frames) -> torch.Tensor:
        """Encode a window rendered just now, through the same codec the training clips went through."""
        return self.encode(through_codec(frames, self.manifest["fps"]))

    def observation_window(self, data: Path, episode_id: str) -> torch.Tensor:
        """The pre-execution window (the prelude) of a recorded episode's clip."""
        frames = clip_frames(data / "clips" / f"{episode_id}.mp4", self.manifest["frames_total"])
        return self.encode(frames[:self.manifest["window_frames"]])

    def plan_chunks(self, trace) -> torch.Tensor:
        """Skill trace -> (steps, 1, window_frames, ACTION_DIM), normalised."""
        actions = compile_plan(trace, self.manifest["phase_frames"], self.manifest["home_waypoint"],
                               self.manifest["frames_total"], self.manifest["prelude_frames"])
        plan = torch.from_numpy(chunks(actions, self.manifest["window_frames"])[1:]).to(self.device)
        return ((plan - self.norm["action_mean"]) / self.norm["action_std"]).unsqueeze(1)

    def verify(self, latent: torch.Tensor, trace) -> VerificationResult:
        with torch.inference_mode():
            final, _ = rollout(self.members, latent, self.plan_chunks(trace))
            probability = success_probability(self.readout, final)
            position, logits = self.readout(final)
            block = (position.mean(0)[0, :2] * self.norm["state_std"][:2] + self.norm["state_mean"][:2]).tolist()
            lifted, in_target = logits.sigmoid().mean(0)[0].tolist()
        mean, spread = float(probability.mean()), float(probability.std())
        failure = "grasp misses the block" if lifted < 0.5 else ("block ends outside the landing zone" if in_target < 0.5 else None)
        suggestion = {"grasp misses the block": "re-aim the grasp at the observed block centre",
                      "block ends outside the landing zone": "move the place target onto the green zone"}.get(failure)
        return VerificationResult(mean >= self.threshold, mean, spread, lifted, in_target, block, failure, suggestion)

    def verify_pick_place(self, latent, block_xy, target_xy, grasp_offset_xy=(0.0, 0.0)) -> VerificationResult:
        return self.verify(latent, pick_place_trace(block_xy, target_xy, grasp_offset_xy))

    def rank(self, latent, candidates: list[dict]) -> list[tuple[dict, VerificationResult]]:
        """Score several proposed plans against one observation, best imagined outcome first."""
        scored = [(candidate, self.verify_pick_place(latent, **candidate)) for candidate in candidates]
        return sorted(scored, key=lambda pair: (-pair[1].success_probability, pair[1].uncertainty))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, help="episode whose prelude window is used as the observation")
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_wm"))
    ap.add_argument("--checkpoint", type=Path, default=Path("models/latent_dynamics.pt"))
    ap.add_argument("--model", type=Path, default=Path("models/vjepa2-vitl-fpc64-256"))
    ap.add_argument("--target-x", type=float); ap.add_argument("--target-y", type=float)
    ap.add_argument("--grasp-offset-x", type=float, default=0.0); ap.add_argument("--grasp-offset-y", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--rank-alternatives", action="store_true", help="also score the recorded plan against two safer variants")
    args = ap.parse_args()

    record = next(json.loads(line) for line in (args.data / "records.jsonl").open()
                  if json.loads(line)["episode_id"] == args.episode)
    block_xy = record["state_before"]["block_pos"][:2]
    target = [args.target_x, args.target_y] if args.target_x is not None else record["skill"]["params"]["target_xy"]
    offset = [args.grasp_offset_x, args.grasp_offset_y]
    if args.target_x is None and offset == [0.0, 0.0]:
        offset = record["skill"]["params"].get("grasp_offset_xy", offset)

    verifier = Verifier(args.checkpoint, args.model, args.threshold)
    latent = verifier.observation_window(args.data, args.episode)
    result = verifier.verify_pick_place(latent, block_xy, target, offset)
    output = {"episode_id": args.episode, "observed_block_xy": [round(v, 4) for v in block_xy],
              "plan": {"target_xy": target, "grasp_offset_xy": offset}, **asdict(result),
              "actual_outcome": {"success": record["outcome"]["success"], "failure_mode": record["outcome"]["failure_mode"]}}
    if args.rank_alternatives:
        candidates = [{"block_xy": block_xy, "target_xy": target, "grasp_offset_xy": offset},
                      {"block_xy": block_xy, "target_xy": target, "grasp_offset_xy": [0.0, 0.0]},
                      {"block_xy": block_xy, "target_xy": record["state_before"]["target_pos"], "grasp_offset_xy": [0.0, 0.0]}]
        output["ranked"] = [{"plan": {k: list(np.round(v, 4)) for k, v in candidate.items()},
                             "success_probability": round(scored.success_probability, 4),
                             "uncertainty": round(scored.uncertainty, 4), "likely_failure": scored.likely_failure}
                            for candidate, scored in verifier.rank(latent, candidates)]
    print(json.dumps(output, indent=2, default=float))


if __name__ == "__main__":
    main()
