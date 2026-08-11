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
from waddle_wm.sim import relling_scene as scene
from waddle_wm.sim.env import TARGET_RADIUS, pick_place_trace
from waddle_wm.train_multiblock_world_model import StateWorldModel, task_features
from waddle_wm.train_latent_dynamics import STATE_DIM, Dynamics, ObjectReadout, Readout, rollout, success_probability


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

    def __init__(self, checkpoint: Path, model: Path = Path("models/vjepa2-vitl-fpc64-256"), threshold: float | None = None, device=None):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.manifest = saved["manifest"]
        self.threshold = saved.get("decision_threshold", 0.5) if threshold is None else threshold
        self.model_path = model
        self.model_type = saved.get("model_type", "latent_dynamics")
        self.norm = {key: value.to(self.device) for key, value in saved["normalization"].items()}
        self._encoder = None
        if self.model_type == "multiblock_state":
            self.members = nn.ModuleList([StateWorldModel(saved["context_dim"], saved["plan_dim"])
                                          for _ in range(saved["member_count"])])
            self.members.load_state_dict(saved["members"]); self.members.to(self.device).eval()
            self.binary_dim, self.object_conditioned = 0, False
            return
        self.members = nn.ModuleList([Dynamics(saved["latent_dim"], saved["chunk_dim"]) for _ in range(saved["member_count"])])
        self.members.load_state_dict(saved["members"]); self.members.to(self.device).eval()
        self.state_dim, self.binary_dim = saved.get("state_dim", STATE_DIM), saved.get("binary_dim", 2)
        self.object_conditioned = saved.get("object_conditioned_readout", False)
        self.readout = (ObjectReadout(saved["latent_dim"], len(saved["manifest"]["block_names"]))
                        if self.object_conditioned else Readout(saved["latent_dim"], self.state_dim, self.binary_dim))
        self.readout.load_state_dict(saved["readout"]); self.readout.to(self.device).eval()

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
        if self.model_type == "multiblock_state":
            return (latent - self.norm["context_mean"]) / self.norm["context_std"]
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

    def verify(self, latent: torch.Tensor, trace, object_name="red_block", destination="green_pad",
               scene_positions=None) -> VerificationResult:
        with torch.inference_mode():
            if self.model_type == "multiblock_state":
                if scene_positions is None:
                    raise ValueError("the multi-block state verifier needs current camera positions")
                names = tuple(self.manifest["block_names"])
                raw_state = torch.tensor([value for name in names for value in scene_positions[name]],
                                         dtype=torch.float32, device=self.device).unsqueeze(0)
                source = names.index(object_name)
                source_xyz = raw_state[0, source * 3:source * 3 + 3]
                destination_xyz = (torch.tensor(scene.TARGET_POS, device=self.device) if destination == "green_pad" else
                                   raw_state[0, names.index(destination) * 3:names.index(destination) * 3 + 3])
                grasp = torch.tensor(next(e["target"] for e in trace if e["phase"] == "descend"), device=self.device)
                place = torch.tensor(next(e["target"] for e in trace if e["phase"] == "place"), device=self.device)
                plan = torch.cat([grasp - source_xyz, place - destination_xyz]).unsqueeze(0)
                plan = (plan - self.norm["plan_mean"]) / self.norm["plan_std"]
                state = (raw_state - self.norm["state_mean"]) / self.norm["state_std"]
                task = task_features([object_name], [destination], names, self.device)
                outputs = [member(latent, state, plan, task) for member in self.members]
                predicted = torch.stack([output[0] for output in outputs])
                logits = torch.stack([output[1] for output in outputs])
                predicted = predicted * self.norm["state_std"] + self.norm["state_mean"]
                block_xyz = predicted[..., source * 3:source * 3 + 3]
                lifted_score, scores = logits[..., 0].sigmoid(), logits[..., 1].sigmoid()
                mean, spread = float(scores.mean()), float(scores.std())
                lifted = float(lifted_score.mean())
                in_target = min(1.0, mean / max(lifted, 1e-6))
                block = block_xyz.mean(0)[0, :2].tolist()
                failure = "grasp misses the block" if lifted < 0.5 else ("block misses the destination" if in_target < 0.5 else None)
                suggestion = {"grasp misses the block": "re-aim the grasp at the observed block centre",
                              "block misses the destination": "move the place target onto the destination"}.get(failure)
                return VerificationResult(mean >= self.threshold, mean, spread, lifted, in_target, block, failure, suggestion)
            final, _ = rollout(self.members, latent, self.plan_chunks(trace))
            if not self.binary_dim:
                names = self.manifest["block_names"]
                source = names.index(object_name)
                position = self.readout(final, source)
                initial = self.readout(latent, source)
                initial = initial * self.norm["state_std"] + self.norm["state_mean"]
                block_xyz = position * self.norm["state_std"] + self.norm["state_mean"]
                before_xyz = initial
                moved = (block_xyz[..., :2] - before_xyz[..., :2]).norm(dim=-1)
                lifted_score = torch.sigmoid((moved - 0.025) / 0.01)
                if destination == "green_pad":
                    target = torch.tensor(next(e["target"][:2] for e in trace if e["phase"] == "place"),
                                          device=self.device)
                    distance = (block_xyz[..., :2] - target).norm(dim=-1)
                    target_score = torch.sigmoid((TARGET_RADIUS - distance) / 0.02)
                else:
                    target_index = names.index(destination)
                    target_xyz = (self.readout(final, target_index) * self.norm["state_std"] +
                                  self.norm["state_mean"])
                    distance = (block_xyz[..., :2] - target_xyz[..., :2]).norm(dim=-1)
                    vertical = block_xyz[..., 2] - target_xyz[..., 2]
                    target_score = torch.minimum(torch.sigmoid((0.027 - distance) / 0.008),
                                                 torch.sigmoid((vertical - 0.027) / 0.006))
                scores = torch.minimum(lifted_score, target_score)
                mean, spread = float(scores.mean()), float(scores.std())
                lifted, in_target = float(lifted_score.mean()), float(target_score.mean())
                block = block_xyz.mean(0)[0, :2].tolist()
                failure = "grasp misses the block" if lifted < 0.5 else ("block misses the destination" if in_target < 0.5 else None)
                suggestion = {"grasp misses the block": "re-aim the grasp at the observed block centre",
                              "block misses the destination": "move the place target onto the destination"}.get(failure)
                return VerificationResult(mean >= self.threshold, mean, spread, lifted, in_target,
                                          block, failure, suggestion)
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
