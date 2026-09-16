"""Analytic evaluation and input-boundary checks; no model fitting required."""
import contextlib
import csv
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hierarchical.data import Record
import hierarchical.train as training


class TableModel(nn.Module):
    """Return known normalized hierarchical probabilities by example index."""

    def __init__(self):
        super().__init__()
        joint = torch.tensor([
            [.65, .15, .10, .10],
            [.30, .10, .50, .10],
            [.40, .05, .30, .25],
            [.10, .70, .10, .10],
            [.50, .10, .30, .10],
        ], dtype=torch.float64)
        gate = torch.stack((joint[:, :2].sum(1), joint[:, 2:].sum(1)), dim=1)
        self.register_buffer('gate', gate.log())
        self.register_buffer('log_probs', joint.log())
        self.register_buffer('parents', torch.tensor([0, 0, 1, 1]))

    def forward(self, images):
        indices = images[:, 0].long()
        return self.gate[indices], self.log_probs[indices]


def analytic_loader(batch_size):
    return DataLoader(TensorDataset(
        torch.arange(5).reshape(5, 1),
        torch.tensor([0, 0, 0, 0, 1]),
        torch.tensor([0, 0, 0, 1, 2]),
    ), batch_size=batch_size, shuffle=False)


def arguments(root, **changes):
    values = dict(data_dir=root, output=root / 'run', weights_path=None,
                  device='cpu', epochs=1, batch_size=2, learning_rate=1e-4, seed=7)
    values.update(changes)
    return SimpleNamespace(**values)


class TrainingValidationTests(unittest.TestCase):
    def test_evaluation_rejects_nonfinite_outputs_before_loss_or_argmax(self):
        for output in ('gate', 'log_probs'):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(output=output, value=value):
                    model = TableModel()
                    getattr(model, output)[0, 0] = value
                    with (
                        patch.object(training, 'classification_loss',
                                     side_effect=AssertionError('invalid output reached loss')) as loss,
                        patch.object(torch.Tensor, 'argmax',
                                     side_effect=AssertionError('invalid output reached argmax')),
                        self.assertRaisesRegex(RuntimeError, 'nonfinite evaluation output'),
                    ):
                        training.evaluate(model, analytic_loader(2), torch.device('cpu'))
                    loss.assert_not_called()

    def test_evaluation_rejects_nonfinite_loss_before_argmax(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with (
                    patch.object(training, 'classification_loss', return_value=torch.tensor(value)),
                    patch.object(torch.Tensor, 'argmax',
                                 side_effect=AssertionError('invalid loss reached argmax')),
                    self.assertRaisesRegex(RuntimeError, 'nonfinite evaluation loss'),
                ):
                    training.evaluate(TableModel(), analytic_loader(2), torch.device('cpu'))

    def test_evaluation_weights_loss_and_macro_recall_across_unequal_batches(self):
        # Class frequencies are 3/1/1, with recalls 2/3, 1 and 0. Each
        # row's loss is superclass cross-entropy plus joint subclass NLL.
        expected_loss = sum(-math.log(parent) - math.log(child) for parent, child in [
            (.8, .65), (.4, .3), (.45, .4), (.8, .7), (.4, .3),
        ]) / 5
        for batch_size in (1, 2, 3, 5):
            with self.subTest(batch_size=batch_size):
                result = training.evaluate(TableModel(), analytic_loader(batch_size),
                                           torch.device('cpu'))
                self.assertEqual(result['examples'], 5)
                self.assertAlmostEqual(result['loss'], expected_loss, places=12)
                self.assertAlmostEqual(result['subclass_accuracy'], 3 / 5)
                self.assertAlmostEqual(result['subclass_macro_recall'], 5 / 9)
                self.assertAlmostEqual(result['gate_superclass_accuracy'], 2 / 5)
                self.assertAlmostEqual(result['coherent_superclass_accuracy'], 3 / 5)

    def test_nonfinite_and_nonpositive_learning_rates_fail_before_data_access(self):
        for learning_rate in (math.nan, math.inf, -math.inf, 0.0, -1.0):
            with self.subTest(learning_rate=learning_rate):
                with tempfile.TemporaryDirectory() as directory:
                    with (
                        patch.object(training, 'load_records',
                                     side_effect=AssertionError('invalid rate reached data loading')) as load,
                        self.assertRaisesRegex(ValueError, 'finite learning_rate > 0 required'),
                    ):
                        training.train(arguments(Path(directory), learning_rate=learning_rate))
                    load.assert_not_called()

    def test_minimum_image_groups_fail_clearly_before_model_or_loader_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train_images').mkdir()
            with (root / 'train_data.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['image', 'superclass_index', 'subclass_index'])
                for index in range(3):
                    name = f'{index}.png'
                    Image.new('RGB', (8, 8), (index, 0, 0)).save(root / 'train_images' / name)
                    writer.writerow([name, 5, 10])
            with (
                patch.object(training, 'ImageDataset',
                             side_effect=AssertionError('singleton split reached dataset construction')) as dataset,
                patch.object(training, 'DataLoader',
                             side_effect=AssertionError('singleton split reached loader construction')) as loader,
                patch.object(training, 'SoftGatedExpertModel',
                             side_effect=AssertionError('singleton split reached model construction')) as model,
                self.assertRaisesRegex(ValueError, 'at least two training examples are required'),
            ):
                training.train(arguments(root))
            dataset.assert_not_called()
            loader.assert_not_called()
            model.assert_not_called()

    def test_nonfinite_report_is_rejected_before_checkpoint_write(self):
        # Mock the whole fitting path: this checks persistence ordering without
        # constructing a real model, reading images, or updating parameters.
        records = [Record(f'{index}.png', 0, 0, f'pixels-{index}') for index in range(4)]
        parts = {'train': records[:2], 'validation': records[2:3], 'test': records[3:]}
        metrics = dict(examples=1, loss=.5, subclass_accuracy=1.0,
                       subclass_macro_recall=1.0, gate_superclass_accuracy=1.0,
                       coherent_superclass_accuracy=1.0)
        batch = (torch.zeros(2, 1), torch.zeros(2, dtype=torch.long),
                 torch.zeros(2, dtype=torch.long))
        for value in (math.nan, math.inf):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = Mock()
                model.parameters.return_value = []
                model.return_value = (torch.zeros(2, 1), torch.zeros(2, 1))
                model.state_dict.return_value = {'weight': torch.zeros(1)}
                with (
                    patch.object(training, 'load_records', return_value=(records, [0], [5], [10])),
                    patch.object(training, 'split_records', return_value=parts),
                    patch.object(training, 'ImageDataset', return_value=[]),
                    patch.object(training, 'WeightedRandomSampler'),
                    patch.object(training, 'DataLoader', side_effect=[[batch], [], []]),
                    patch.object(training, 'SoftGatedExpertModel', return_value=model),
                    patch.object(torch.optim, 'Adam'),
                    patch.object(training, 'classification_loss', return_value=torch.tensor(0.0)),
                    patch.object(torch.Tensor, 'backward'),
                    patch.object(training, 'evaluate', side_effect=[metrics, {**metrics, 'loss': value}]),
                    patch.object(torch, 'save') as save,
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaisesRegex(ValueError, 'Out of range float'),
                ):
                    training.train(arguments(root))
                save.assert_not_called()
                self.assertFalse((root / 'run' / 'best.pt').exists())
                self.assertFalse((root / 'run' / 'metrics.json').exists())


if __name__ == '__main__':
    unittest.main()
