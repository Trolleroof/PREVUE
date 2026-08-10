"""Small runnable check for the action-conditioned latent modules."""
import torch

from waddle_wm.train_action_conditioned_latent import HEADS, VARIANTS, LatentPredictor, OutcomeHead, dataset_audit, sample_window


def record(episode_id, split, target_xy, success):
    return {"episode_id": episode_id, "split": split, "skill": {"params": {"target_xy": target_xy}},
            "state_before": {"block_pos": [0.38, -0.18, 0.018]},
            "state_after": {"block_pos": [*target_xy, 0.018], "target_pos": [0.5, 0.3], "max_block_z": 0.27},
            "outcome": {"success": success, "failure_mode": None if success else "target_miss"}}


def main():
    batch, dim = 4, 1024
    predictor = LatentPredictor(dim).eval()  # dropout off: every reported metric is an eval-mode number
    head = OutcomeHead(dim)
    context = torch.randn(batch, dim)
    plan = torch.randn(batch, 2)
    predicted = predictor(context, plan)
    decoded = head(predicted)
    assert predicted.shape == (batch, dim)
    assert decoded.shape == (batch, len(HEADS) + 2)

    # Gating an input off must actually change the prediction, or the ablation baselines are vacuous.
    for name, (use_context, use_action) in VARIANTS.items():
        gated = predictor(context * use_context, plan * use_action)
        assert gated.shape == predicted.shape, name
        assert (name == "context_plus_action") == torch.equal(gated, predicted), name

    # The plan encoder must widen the action so it is not drowned out by the 1024-d context.
    assert predictor.plan_encoder(plan).shape == (batch, 128)
    predictor.train()
    assert not torch.equal(predictor(context, plan), predictor(context, plan)), "context dropout inactive in train mode"
    predictor.eval()
    assert torch.equal(predictor(context, plan), predictor(context, plan)), "context dropout active in eval mode"

    clip = list(range(10))
    assert sample_window(clip, 0, 5, 4) == [0, 1, 3, 4]
    assert sample_window(clip, 5, 10, 4) == [5, 6, 8, 9]
    assert sample_window(clip, 3, 3, 2) == [3, 3]  # empty window falls back instead of crashing

    audit = dataset_audit([record("a", "train", [0.5, 0.3], True), record("b", "val", [0.33, 0.3], False)])
    assert audit["episodes"] == 2 and audit["splits"]["train"]["success"] == 1
    assert audit["identities"]["target_miss_equals_not_success"]
    assert audit["identities"]["success_equals_plan_within_target_radius"]
    assert audit["identities"]["distinct_target_sites"] == 1
    print("action-conditioned latent module check passed")


if __name__ == "__main__":
    main()
