"""ResNet18 gate/experts derived from model2.ipynb; see PROVENANCE.md."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class SoftGatedExpertModel(nn.Module):
    """P(subclass|image) = P(parent|image) P(subclass|parent,image).

    Each subclass belongs to exactly one superclass. Labels are contiguous
    internal IDs; external labels are mapped by the data loader. Novel labels
    with no training observations must not be added as untrained output heads.
    """

    def __init__(self, subclass_to_superclass, hidden_dim=256, weights=None):
        super().__init__()
        parents = list(subclass_to_superclass)
        if not parents or any(type(p) is not int or p < 0 for p in parents):
            raise ValueError("parents must be a nonempty list of nonnegative integers")
        count = max(parents) + 1
        if set(parents) != set(range(count)):
            raise ValueError("superclass IDs must be contiguous and every gate must have an expert")
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        backbone = resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        self.total_sub_classes = len(parents)
        self.num_super_classes = count
        self.register_buffer("parents", torch.tensor(parents, dtype=torch.long))
        self.gate = self._head(count, hidden_dim)
        self.experts = nn.ModuleList()
        for parent in range(count):
            indices = torch.tensor([i for i, p in enumerate(parents) if p == parent], dtype=torch.long)
            self.register_buffer(f"expert_indices_{parent}", indices)
            self.experts.append(self._head(len(indices), hidden_dim))

    def _head(self, outputs, hidden_dim):
        return nn.Sequential(nn.Linear(self.feature_dim, hidden_dim), nn.ReLU(),
                             nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, outputs))

    def forward(self, images):
        features = self.feature_extractor(images).flatten(1)
        gate_logits = self.gate(features)
        gate_log_probs = F.log_softmax(gate_logits, dim=1)
        # Keep the original factorization, but compute it in log space.
        log_probs = features.new_full((images.shape[0], self.total_sub_classes), -torch.inf)
        for parent, expert in enumerate(self.experts):
            conditional = F.log_softmax(expert(features), dim=1)
            indices = getattr(self, f"expert_indices_{parent}")
            log_probs[:, indices] = conditional + gate_log_probs[:, parent:parent + 1]
        return gate_logits, log_probs

    @torch.no_grad()
    def predict(self, images):
        """Return a coherent superclass/subclass pair. Call eval() first."""
        if self.training:
            raise RuntimeError("call model.eval() before prediction")
        _, log_probs = self(images)
        subclass = log_probs.argmax(1)
        return self.parents[subclass], subclass


def classification_loss(gate_logits, sub_log_probs, super_labels, sub_labels):
    # Retains the notebook's equal weighting of superclass and joint subclass
    # loss: the parent signal is present in both terms, not only the first.
    return F.cross_entropy(gate_logits, super_labels) + F.nll_loss(sub_log_probs, sub_labels)
