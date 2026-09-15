"""Offline forward/backward check. Random tensors are not an accuracy benchmark."""
import json
import torch
from .model import SoftGatedExpertModel, classification_loss


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = SoftGatedExpertModel([0, 1, 0, 1], hidden_dim=16)
    images = torch.randn(4, 3, 64, 64)
    labels = torch.tensor([0, 1, 2, 3])
    gate, log_probs = model(images)
    loss = classification_loss(gate, log_probs, model.parents[labels], labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    parent, child = model.predict(images)
    print(json.dumps({"demo": "random tensors; no accuracy claim", "loss": loss.item(),
                      "finite_gradients": True, "superclasses": parent.tolist(),
                      "subclasses": child.tolist()}))


if __name__ == '__main__':
    main()
