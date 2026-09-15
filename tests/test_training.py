import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import torch
from hierarchical.model import SoftGatedExpertModel
from hierarchical.train import train


class TrainingTests(unittest.TestCase):
    def test_complete_cpu_pipeline_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'train_images'
            images.mkdir()
            rows = []
            for label in range(4):
                for sample in range(6):
                    name = f'{label}-{sample}.png'
                    Image.new('RGB', (16, 16), (label * 50, sample * 30, 80)).save(images / name)
                    rows.append({'image': name, 'superclass_index': label // 2, 'subclass_index': label})
            with (root / 'train_data.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['image', 'superclass_index', 'subclass_index'])
                writer.writeheader()
                writer.writerows(rows)
            args = SimpleNamespace(data_dir=root, output=root / 'run', weights_path=None,
                                   device='cpu', epochs=1, batch_size=8, learning_rate=1e-4, seed=7)
            report = train(args)
            self.assertEqual(report['best_epoch'], 1)
            self.assertTrue(all(v == 0 for v in report['pixel_group_overlap'].values()))
            self.assertEqual(report['test']['examples'], 4)
            self.assertEqual(sum(v['examples'] for v in report['splits'].values()), 24)
            checkpoint = torch.load(root / 'run/best.pt', weights_only=True, map_location='cpu')
            model = SoftGatedExpertModel(checkpoint['hierarchy']).eval()
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            with torch.no_grad():
                _, log_probs = model(torch.zeros(1, 3, 64, 64))
            self.assertTrue(torch.isfinite(log_probs).all())
            with self.assertRaises(ValueError):
                train(args)  # Existing outputs must not be silently overwritten.


if __name__ == '__main__':
    unittest.main()
