"""Train an action-conditioned latent dynamics model on frozen V-JEPA windows.

    z_hat_{k+1} = z_k + f(z_k, action chunk over window k+1)     <- the world model
    block xyz, pinch xyz, lifted, in_target = g(z)               <- grounded readout

`g` only ever sees real latents during training; verification decodes an imagined
latent with it, so the reported rollout numbers are not self-fulfilling.
Uncertainty is the disagreement of an ensemble of independently initialised `f`s.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from waddle_wm import windows
from waddle_wm.actions import ACTION_DIM

STATE_DIM, BINARY_DIM = 6, 2  # block xyz + pinch xyz, then lifted + in_target


class Dynamics(nn.Module):
    """One ensemble member: residual next-latent prediction from a latent and an action chunk."""

    def __init__(self, latent_dim: int, chunk_dim: int, hidden: int = 512, action_hidden: int = 128):
        super().__init__()
        self.action = nn.Sequential(nn.Flatten(1), nn.Linear(chunk_dim, action_hidden), nn.GELU(),
                                    nn.Linear(action_hidden, action_hidden), nn.GELU())
        self.net = nn.Sequential(nn.Linear(latent_dim + action_hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent_dim))

    def forward(self, latent, chunk):
        return latent + self.net(torch.cat([latent, self.action(chunk)], dim=-1))


class Readout(nn.Module):
    """Grounded state decoded from a latent, real or imagined."""

    def __init__(self, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, STATE_DIM + BINARY_DIM))

    def forward(self, latent):
        out = self.net(latent)
        return out[..., :STATE_DIM], out[..., STATE_DIM:]


def rollout(members, latent, chunk_sequence):
    """Chain the ensemble forward over a (steps, batch, window, action) plan.

    Returns (members, batch, latent) at the final step and the per-step mean latent.
    """
    state = latent.unsqueeze(0).expand(len(members), *latent.shape)
    trace = []
    for chunk in chunk_sequence:
        state = torch.stack([member(state[i], chunk) for i, member in enumerate(members)])
        trace.append(state)
    return state, trace


def success_probability(readout, latents):
    """P(success) = P(lifted) * P(in_target) decoded from a latent, per ensemble member."""
    _, logits = readout(latents)
    probability = logits.sigmoid()
    return probability[..., 0] * probability[..., 1]


def label_audit(data, manifest):
    """Check the dataset actually supports the task before anything is trained."""
    terminal = data["state"][:, -1]
    lifted, in_target = terminal[:, 6].astype(bool), terminal[:, 7].astype(bool)
    modes, counts = np.unique(data["failure_mode"], return_counts=True)
    audit = {"episodes": len(data["success"]), "windows": manifest["windows"], "transitions_per_episode": data["steps"],
             "failure_modes": dict(zip(modes.tolist(), counts.tolist())),
             "splits": {split: int((data["splits"] == split).sum()) for split in sorted(set(data["splits"].tolist()))},
             "terminal_identity_holds": bool(np.all((lifted & in_target) == data["success"].astype(bool))),
             "success_rate": float(data["success"].mean()), "grasp_failure_rate": float((~lifted).mean())}
    return audit


def plan_only_baseline(plan, data, splits):
    """How far a model gets from the proposed plan alone, with no view of the scene.

    The control that matters. `in_target` is decidable from the plan; `lifted`
    should not be, since it depends on where the block actually is, so the gap on
    `lifted` is the part of the verifier that the latent is paying for.
    """
    from sklearn.ensemble import RandomForestClassifier
    features = plan["action"].reshape(len(plan["action"]), -1)
    train, test = splits == "train", splits == "test"
    terminal = data["state"][:, -1]
    result = {"model": "random forest on the flattened compiled plan", "features": int(features.shape[1])}
    for name, target in (("success", data["success"]), ("lifted", terminal[:, 6]), ("in_target", terminal[:, 7])):
        forest = RandomForestClassifier(n_estimators=300, random_state=0).fit(features[train], target[train])
        result[f"test_{name}_accuracy"] = float((forest.predict(features[test]) == target[test]).mean())
        result[f"test_{name}_majority_class"] = float(max(target[test].mean(), 1 - target[test].mean()))
    return result


def normalise(tensor, mask, dim=0):
    mean, std = tensor[mask].mean(dim), tensor[mask].std(dim).clamp_min(1e-6)
    return mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ur5e_wm"))
    ap.add_argument("--embeddings", type=Path, default=None, help="default: <data>/window_embeddings.pt")
    ap.add_argument("--out", type=Path, default=Path("models/latent_dynamics.pt"))
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--rollout-weight", type=float, default=1.0, help="weight on the free-running rollout loss")
    ap.add_argument("--focus-step", type=int, default=2, help="0-based transition to upweight; 2 is z2 -> z3 close/lift")
    ap.add_argument("--focus-weight", type=float, default=3.0, help="loss multiplier for --focus-step")
    ap.add_argument("--grounding-weight", type=float, default=0.0,
                    help="weight on decoding the rollout with the detached readout; 0.2 and 1.0 both hurt on this corpus")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    manifest = json.loads((args.data / "manifest.json").read_text())
    records = [json.loads(line) for line in (args.data / "records.jsonl").open()]
    if args.max_episodes:
        records = records[:args.max_episodes]
    cache = torch.load(args.embeddings or args.data / "window_embeddings.pt", weights_only=False)

    data = windows.build(records, manifest)
    plan = windows.build(records, manifest, planned=True)
    audit = label_audit(data, manifest)
    print(json.dumps({"label_audit": audit}, indent=2), flush=True)

    latents = torch.stack([cache[episode] for episode in data["episode_ids"]]).float()   # (E, windows, D)
    action = torch.from_numpy(data["action"])                                            # (E, steps, W, A)
    planned_action = torch.from_numpy(plan["action"])
    state = torch.from_numpy(data["state"])                                              # (E, windows, 8)
    success = torch.from_numpy(data["success"])
    episodes, window_count, latent_dim = latents.shape
    steps = data["steps"]

    split = {name: torch.from_numpy(data["splits"] == name) for name in ("train", "val", "test")}
    if not all(mask.any() for mask in split.values()):
        raise ValueError("records need non-empty train, val, and test splits")

    latent_mean, latent_std = normalise(latents.reshape(-1, latent_dim), split["train"].repeat_interleave(window_count))
    action_mean, action_std = normalise(action.reshape(-1, ACTION_DIM), split["train"].repeat_interleave(steps * manifest["window_frames"]))
    state_mean, state_std = normalise(state.reshape(-1, STATE_DIM + BINARY_DIM)[:, :STATE_DIM], split["train"].repeat_interleave(window_count))
    latents = (latents - latent_mean) / latent_std
    action, planned_action = (action - action_mean) / action_std, (planned_action - action_mean) / action_std
    state = torch.cat([(state[..., :STATE_DIM] - state_mean) / state_std, state[..., STATE_DIM:]], dim=-1)

    latents, action, planned_action, state, success = (t.to(device) for t in (latents, action, planned_action, state, success))
    state_mean, state_std = state_mean.to(device), state_std.to(device)
    split = {name: mask.to(device) for name, mask in split.items()}

    chunk_dim = manifest["window_frames"] * ACTION_DIM
    members = nn.ModuleList([Dynamics(latent_dim, chunk_dim) for _ in range(args.members)]).to(device)
    readout = Readout(latent_dim).to(device)
    opt = torch.optim.AdamW([*members.parameters(), *readout.parameters()], lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(0)
    train_index = split["train"].nonzero().flatten()
    bootstrap = [train_index[torch.randint(len(train_index), (len(train_index),), generator=generator)].to(device)
                 for _ in range(args.members)]

    def latent_error(predicted, target):
        return nn.functional.mse_loss(predicted, target) + (1 - nn.functional.cosine_similarity(predicted, target, dim=-1)).mean()

    step_weights = torch.ones(steps, device=device)
    if 0 <= args.focus_step < steps:
        step_weights[args.focus_step] = args.focus_weight
    step_weights = step_weights / step_weights.mean()

    def transition_error(predicted, target):
        predicted = predicted.reshape(-1, steps, latent_dim)
        target = target.reshape(-1, steps, latent_dim)
        mse = (predicted - target).pow(2).mean(-1)
        cosine = 1 - nn.functional.cosine_similarity(predicted, target, dim=-1)
        return ((mse + cosine) * step_weights).mean()

    def frozen_readout(latent):
        """`readout` with its parameters detached: gradients reach the dynamics, never the head.

        This is what keeps the grounding term from being self-fulfilling — the
        readout is still fitted on real latents only, so it cannot adapt itself to
        whatever the rollout happens to produce.
        """
        parameters = {name: value.detach() for name, value in readout.named_parameters()}
        return torch.func.functional_call(readout, parameters, (latent,))

    def grounding_error(latent, target):
        predicted, logits = frozen_readout(latent)
        return (nn.functional.mse_loss(predicted, target[:, :STATE_DIM])
                + nn.functional.binary_cross_entropy_with_logits(logits, target[:, STATE_DIM:]))

    def dynamics_loss(mask_or_index, per_member_index=None):
        """One-step teacher forcing, the free-running rollout, and what the rollout decodes to.

        Latent regression alone optimises the bulk of the latent — arm pose, which
        moves the same way whether or not the grasp caught the block. The grounding
        term is what forces the imagined future to keep the bit the verifier reads.
        """
        teacher, free, grounded = 0.0, 0.0, 0.0
        for i, member in enumerate(members):
            index = per_member_index[i] if per_member_index is not None else mask_or_index
            teacher = teacher + transition_error(member(latents[index, :-1].reshape(-1, latent_dim),
                                                        action[index].reshape(-1, manifest["window_frames"], ACTION_DIM)),
                                                 latents[index, 1:].reshape(-1, latent_dim))
            imagined = latents[index, 0]
            for k in range(steps):
                imagined = member(imagined, action[index, k])
                free = free + step_weights[k] * latent_error(imagined, latents[index, k + 1])
                grounded = grounded + step_weights[k] * grounding_error(imagined, state[index, k + 1])
        return (teacher + (args.rollout_weight * free + args.grounding_weight * grounded) / step_weights.sum()) / len(members)

    def readout_loss(index):
        predicted, logits = readout(latents[index].reshape(-1, latent_dim))
        target = state[index].reshape(-1, STATE_DIM + BINARY_DIM)
        return (nn.functional.mse_loss(predicted, target[:, :STATE_DIM])
                + nn.functional.binary_cross_entropy_with_logits(logits, target[:, STATE_DIM:]))

    val_index = split["val"].nonzero().flatten()
    best = {"dynamics": (float("inf"), 0, None), "readout": (float("inf"), 0, None)}
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad()
        (dynamics_loss(None, bootstrap) + readout_loss(train_index)).backward()
        opt.step()
        with torch.inference_mode():
            scores = {"dynamics": float(dynamics_loss(val_index)), "readout": float(readout_loss(val_index))}
        for name, module in (("dynamics", members), ("readout", readout)):   # they overfit at different rates
            if scores[name] < best[name][0]:
                best[name] = (scores[name], epoch, {k: v.clone() for k, v in module.state_dict().items()})
        if epoch % 200 == 0:
            print(f"epoch {epoch}: val dynamics {scores['dynamics']:.4f} (best {best['dynamics'][0]:.4f} @ {best['dynamics'][1]}) "
                  f"readout {scores['readout']:.4f} (best {best['readout'][0]:.4f} @ {best['readout'][1]})", flush=True)
    members.load_state_dict(best["dynamics"][2]); readout.load_state_dict(best["readout"][2])
    members.eval(); readout.eval()

    def evaluate(mask, chunks_source, name):
        index = mask.nonzero().flatten()
        with torch.inference_mode():
            plan_chunks = [chunks_source[index, k] for k in range(steps)]
            final, trace = rollout(members, latents[index, 0], plan_chunks)
            per_step_cosine = [float(nn.functional.cosine_similarity(step.mean(0), latents[index, k + 1], dim=-1).mean())
                               for k, step in enumerate(trace)]
            one_step = torch.stack([member(latents[index, :-1].reshape(-1, latent_dim),
                                           chunks_source[index].reshape(-1, manifest["window_frames"], ACTION_DIM)) for member in members]).mean(0)
            target = latents[index, 1:].reshape(-1, latent_dim)
            scores = success_probability(readout, final)
            probability, uncertainty = scores.mean(0), scores.std(0)
            truth = success[index]
            verdict = (probability >= 0.5).float()
            position, logits = readout(final)
            position, heads = position.mean(0), logits.sigmoid().mean(0)
            true_position, _ = readout(latents[index, -1])
            block_error = ((position[:, :2] - state[index, -1, :2]) * state_std[:2]).norm(dim=-1)
            oracle_error = ((true_position[:, :2] - state[index, -1, :2]) * state_std[:2]).norm(dim=-1)
            correct = verdict == truth
            # the imagined block sits between "still where it started" and "on the target", so the
            # regression is only meaningful where the model has committed to an outcome
            confident = (probability > 0.8) | (probability < 0.2)
            return {
                "episodes": int(mask.sum()),
                "one_step_latent_cosine": float(nn.functional.cosine_similarity(one_step, target, dim=-1).mean()),
                "persistence_baseline_cosine": float(nn.functional.cosine_similarity(latents[index, :-1].reshape(-1, latent_dim), target, dim=-1).mean()),
                "rollout_latent_cosine_per_step": [round(value, 4) for value in per_step_cosine],
                "imagined_block_xy_rmse_m": float(block_error.pow(2).mean().sqrt()),
                "imagined_block_xy_rmse_m_when_confident": float(block_error[confident].pow(2).mean().sqrt()) if confident.any() else None,
                "confident_fraction": float(confident.float().mean()),
                "readout_oracle_block_xy_rmse_m": float(oracle_error.pow(2).mean().sqrt()),
                "readout_oracle_lifted_accuracy": float(((readout(latents[index, -1])[1][:, 0].sigmoid() >= 0.5).float() == state[index, -1, 6]).float().mean()),
                "success_accuracy": float(correct.float().mean()),
                "lifted_accuracy": float(((heads[:, 0] >= 0.5).float() == state[index, -1, 6]).float().mean()),
                "in_target_accuracy": float(((heads[:, 1] >= 0.5).float() == state[index, -1, 7]).float().mean()),
                "brier": float((probability - truth).pow(2).mean()),
                "false_accept_rate": float(((verdict == 1) & (truth == 0)).float().sum() / max(1, int((truth == 0).sum()))),
                "false_reject_rate": float(((verdict == 0) & (truth == 1)).float().sum() / max(1, int((truth == 1).sum()))),
                "mean_uncertainty_when_right": float(uncertainty[correct].mean()) if correct.any() else None,
                "mean_uncertainty_when_wrong": float(uncertainty[~correct].mean()) if (~correct).any() else None,
                "chunks": name,
            }

    result = {"label_audit": audit, "plan_only_baseline": plan_only_baseline(plan, data, data["splits"]),
              "best_dynamics_epoch": best["dynamics"][1], "best_dynamics_val_loss": best["dynamics"][0],
              "best_readout_epoch": best["readout"][1], "best_readout_val_loss": best["readout"][0],
              "rollout_weight": args.rollout_weight, "grounding_weight": args.grounding_weight, "members": args.members,
              "focus_step": args.focus_step, "focus_weight": args.focus_weight,
              "latent_dim": latent_dim, "rollout_steps": steps}
    for name, source in (("executed", action), ("compiled_plan", planned_action)):
        for split_name in ("train", "val", "test"):
            result[f"{split_name}_{name}"] = evaluate(split[split_name], source, name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"members": members.cpu().state_dict(), "readout": readout.cpu().state_dict(),
                "latent_dim": latent_dim, "chunk_dim": chunk_dim, "member_count": args.members,
                "manifest": manifest, "state_keys": list(windows.STATE_KEYS) + list(windows.BINARY_KEYS),
                "normalization": {"latent_mean": latent_mean, "latent_std": latent_std, "action_mean": action_mean,
                                  "action_std": action_std, "state_mean": state_mean.cpu(), "state_std": state_std.cpu()},
                "metrics": result}, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
