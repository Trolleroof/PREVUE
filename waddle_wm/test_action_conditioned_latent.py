"""Small runnable check for the action-conditioned latent modules."""
import torch

from waddle_wm.train_action_conditioned_latent import HEADS, LatentPredictor, OutcomeHead


def main():
    batch, dim = 4, 1024
    predictor = LatentPredictor(dim)
    head = OutcomeHead(dim)
    context = torch.randn(batch, dim)
    plan = torch.randn(batch, 2)
    predicted = predictor(context, plan)
    decoded = head(predicted)
    assert predicted.shape == (batch, dim)
    assert decoded.shape == (batch, len(HEADS) + 2)
    print("action-conditioned latent module check passed")


if __name__ == "__main__":
    main()
