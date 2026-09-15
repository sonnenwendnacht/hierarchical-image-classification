import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from hierarchical.model import SoftGatedExpertModel, classification_loss

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = SoftGatedExpertModel([0, 1, 0, 1], hidden_dim=16)
        self.images = torch.randn(4, 3, 64, 64)

    def test_probabilities_and_parent_marginals(self):
        self.model.eval()
        gate, log_probs = self.model(self.images)
        probs = log_probs.exp()
        torch.testing.assert_close(probs.sum(1), torch.ones(4))
        for parent in range(2):
            torch.testing.assert_close(probs[:, self.model.parents == parent].sum(1),
                                       gate.softmax(1)[:, parent])

    def test_finite_gradients_all_heads(self):
        gate, log_probs = self.model(self.images)
        children = torch.arange(4)
        loss = classification_loss(gate, log_probs, self.model.parents[children], children)
        loss.backward()
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_extreme_logits_stay_finite(self):
        self.model.eval()
        with torch.no_grad():
            self.model.gate[-1].bias.copy_(torch.tensor([1e4, -1e4]))
        gate, log_probs = self.model(self.images)
        labels = torch.tensor([0, 1, 2, 3])
        loss = classification_loss(gate, log_probs, self.model.parents[labels], labels)
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.isfinite(loss))

    def test_coherent_predictions_and_singleton_eval(self):
        self.model.eval()
        parent, child = self.model.predict(self.images[:1])
        self.assertEqual(len(parent), 1)
        torch.testing.assert_close(parent, self.model.parents[child])

    def test_predict_rejects_training_mode(self):
        with self.assertRaises(RuntimeError):
            self.model.predict(self.images)

    def test_hierarchy_validation(self):
        for parents in [[], [-1, 0], [0, 2], [False, 1], [0, 1.0]]:
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                SoftGatedExpertModel(parents)

    def test_state_roundtrip_includes_mapping(self):
        self.model.eval()
        other = SoftGatedExpertModel([0, 1, 0, 1], hidden_dim=16).eval()
        other.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(self.model(self.images), other(self.images))
        self.assertIn('expert_indices_0', self.model.state_dict())

    def test_matches_unchanged_historical_model_on_known_classes(self):
        source = (ROOT / 'historical/model2_class.py').read_text()
        namespace = {'torch': torch, 'nn': nn, 'F': F}
        # Only replace the pretrained download with a random ResNet of the same
        # shape. Copy identical parameters into both implementations below.
        with patch('torchvision.models.resnet18', side_effect=lambda **kwargs: resnet18(weights=None)):
            exec(compile(source, 'historical/model2_class.py', 'exec'), namespace)
            old = namespace['SoftGatedExpertModel'](2, pd.DataFrame({'superclass_index': [0, 1, 0, 1]}), hidden_dim=16)
        old.load_state_dict({k: v for k, v in self.model.state_dict().items() if k in old.state_dict()})
        self.model.eval()
        old.eval()
        new_gate, new_logs = self.model(self.images)
        old_gate, old_probs = old(self.images)
        torch.testing.assert_close(new_gate, old_gate)
        torch.testing.assert_close(new_logs.exp(), old_probs, atol=1e-6, rtol=1e-5)

    def test_historical_exports_match_manifest_and_have_no_outputs(self):
        manifest = json.loads((ROOT / 'historical/manifest.json').read_text())
        for name in ('pc', 'laptop'):
            record = manifest[name]
            raw = (ROOT / 'historical' / record['export_file']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record['export_sha256'])
            notebook = json.loads(raw)
            for cell in notebook['cells']:
                self.assertFalse(cell.get('outputs'))
                self.assertFalse(cell.get('attachments'))
        raw = (ROOT / 'historical/model2_class.py').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest['model_class']['sha256'])


if __name__ == '__main__':
    unittest.main()
