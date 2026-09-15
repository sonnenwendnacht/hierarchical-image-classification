import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from hierarchical.data import (Record, load_records, split_records, balance_weights,
                               fingerprint, ImageDataset)


def fixtures():
    return [Record(f'{c}-{i}.jpg', c // 2, c, f'pixels-{c}-{i}')
            for c in range(4) for i in range(10)]


class DataTests(unittest.TestCase):
    def test_disjoint_deterministic_stratified_split(self):
        rows = fixtures()
        parts = split_records(rows)
        self.assertEqual(parts, split_records(list(reversed(rows))))
        self.assertEqual(sum(map(len, parts.values())), len(rows))
        for selected in parts.values():
            self.assertEqual({r.subclass for r in selected}, {0, 1, 2, 3})
        for first, second in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            self.assertFalse({r.pixel_hash for r in parts[first]} & {r.pixel_hash for r in parts[second]})
        self.assertNotEqual(parts, split_records(rows, seed=43))

    def test_duplicates_never_cross_partitions(self):
        rows = fixtures()
        first = rows[0]
        rows.append(Record('copy.jpg', first.superclass, first.subclass, first.pixel_hash))
        parts = split_records(rows)
        locations = [name for name, selected in parts.items() for r in selected if r.pixel_hash == first.pixel_hash]
        self.assertEqual(len(locations), 2)
        self.assertEqual(len(set(locations)), 1)

    def test_conflicting_duplicate_labels_rejected(self):
        rows = fixtures() + [Record('copy.jpg', 0, 1, 'pixels-0-0')]
        with self.assertRaises(ValueError):
            split_records(rows)

    def test_small_classes_and_invalid_fractions_rejected(self):
        with self.assertRaises(ValueError):
            split_records(fixtures()[:2])
        for val, test in [(0, .2), (.5, .5), (-1, .2), (.8, .3)]:
            with self.assertRaises(ValueError):
                split_records(fixtures(), val_fraction=val, test_fraction=test)

    def test_equal_sampling_mass_per_class_training_only(self):
        rows = fixtures()
        rows = [r for r in rows if r.subclass != 0 or r.image == '0-0.jpg']
        weights = balance_weights(rows)
        for label in range(4):
            self.assertAlmostEqual(sum(w for r, w in zip(rows, weights) if r.subclass == label), 1)

    def test_fingerprint_order_independent_content_sensitive(self):
        rows = fixtures()
        self.assertEqual(fingerprint(rows), fingerprint(rows[::-1]))
        self.assertNotEqual(fingerprint(rows), fingerprint(rows[:-1]))

    def build_local_data(self, root, duplicate=False, conflicting=False, bad_name=None):
        (root / 'train_images').mkdir()
        rows = []
        for i in range(6):
            name = f'{i}.png'
            color = i if not duplicate or i != 1 else 0
            Image.new('RGB', (8, 8), (color, 0, 0)).save(root / 'train_images' / name)
            rows.append({'image': bad_name if i == 0 and bad_name else name,
                         'superclass_index': 99 if conflicting and i == 1 else 5,
                         'subclass_index': 10})
        with (root / 'train_data.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image', 'superclass_index', 'subclass_index'])
            writer.writeheader()
            writer.writerows(rows)

    def test_load_maps_external_ids_and_hashes_decoded_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_local_data(root, duplicate=True)
            records, parents, supers, subs = load_records(root)
            self.assertEqual((parents, supers, subs), ([0], [5], [10]))
            self.assertEqual(records[0].pixel_hash, records[1].pixel_hash)

    def test_bad_hierarchy_and_path_traversal_rejected(self):
        for kwargs in [{'conflicting': True}, {'bad_name': '../escape.png'}, {'bad_name': '/tmp/escape.png'}]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.build_local_data(root, **kwargs)
                with self.assertRaises(ValueError):
                    load_records(root)

    def test_validation_transform_repeatable_and_missing_image_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_local_data(root)
            records, _, _, _ = load_records(root)
            dataset = ImageDataset(root / 'train_images', records)
            torch.testing.assert_close(dataset[0][0], dataset[0][0])
            missing = ImageDataset(root / 'train_images', [Record('missing.png', 0, 0, 'unused')])
            with self.assertRaises(FileNotFoundError):
                missing[0]


if __name__ == '__main__':
    unittest.main()
