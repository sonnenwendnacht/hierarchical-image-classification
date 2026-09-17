"""Output ownership checks with tiny artifacts and no real-data fitting."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import hierarchical.train as training
import verify_reproduction


def training_args(root):
    return SimpleNamespace(data_dir=root, output=root / 'nested' / 'run',
                           weights_path=None, device='cpu', epochs=1,
                           batch_size=2, learning_rate=1e-4, seed=7)


def tiny_runs(root):
    """Valid comparator inputs; one tensor is enough to test file handling."""
    first, second = root / 'first', root / 'second'
    source = Path(training.__file__)
    report = {
        'source_sha256': {'hierarchical/train.py': hashlib.sha256(source.read_bytes()).hexdigest()},
        'best_epoch': 1,
        'config': {'seed': 7},
    }
    checkpoint = {'state_dict': {'weight': torch.zeros(1)}, 'hierarchy': [0],
                  'superclass_ids': [5], 'subclass_ids': [10], 'best_epoch': 1}
    for directory in (first, second):
        directory.mkdir()
        (directory / 'metrics.json').write_text(json.dumps(report))
        torch.save(checkpoint, directory / 'best.pt')
    originals = {directory / name: (directory / name).read_bytes()
                 for directory in (first, second) for name in ('metrics.json', 'best.pt')}
    return first, second, originals


def run_verifier(first, second, output=None):
    argv = ['verify_reproduction.py', str(first), str(second)]
    if output is not None:
        argv.extend(['--output', str(output)])
    with patch('sys.argv', argv):
        return verify_reproduction.main()


class OutputSafetyTests(unittest.TestCase):
    def assert_inputs_unchanged(self, originals):
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content, str(path))

    def assert_parser_error(self, first, second, output):
        stderr = io.StringIO()
        with (contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught):
            run_verifier(first, second, output)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn('error:', stderr.getvalue())

    def test_training_refuses_existing_empty_directory_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            args = training_args(Path(directory))
            args.output.mkdir(parents=True)
            with (patch.object(training, 'load_records') as load,
                  self.assertRaisesRegex(ValueError, 'output directory already exists')):
                training.train(args)
            load.assert_not_called()
            self.assertEqual(list(args.output.iterdir()), [])

    def test_training_reserves_leaf_before_loading_and_retains_failed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            args = training_args(Path(directory))
            calls = []

            def load_with_competing_run(_):
                calls.append(True)
                if len(calls) != 1:
                    self.fail('second run reached data loading in an already reserved directory')
                self.assertTrue(args.output.is_dir())
                with self.assertRaisesRegex(ValueError, 'output directory already exists'):
                    training.train(args)
                raise RuntimeError('controlled failure after reservation')

            with (patch.object(training, 'load_records', side_effect=load_with_competing_run),
                  self.assertRaisesRegex(RuntimeError, 'controlled failure after reservation')):
                training.train(args)
            self.assertEqual(len(calls), 1)
            self.assertTrue(args.output.is_dir())
            self.assertEqual(list(args.output.iterdir()), [])

    def test_verifier_refuses_existing_output_without_changing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, originals = tiny_runs(root)
            output = root / 'existing.json'
            output.write_bytes(b'previous verification record\n')
            self.assert_parser_error(first, second, output)
            self.assertEqual(output.read_bytes(), b'previous verification record\n')
            self.assert_inputs_unchanged(originals)

    def test_verifier_refuses_input_paths_and_aliases(self):
        for kind in ('first_metrics', 'second_metrics', 'first_checkpoint',
                     'second_checkpoint', 'symlink', 'hardlink'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                first, second, originals = tiny_runs(root)
                paths = {'first_metrics': first / 'metrics.json',
                         'second_metrics': second / 'metrics.json',
                         'first_checkpoint': first / 'best.pt',
                         'second_checkpoint': second / 'best.pt'}
                if kind == 'symlink':
                    output = root / 'alias.json'
                    output.symlink_to(first / 'metrics.json')
                elif kind == 'hardlink':
                    output = root / 'alias.pt'
                    output.hardlink_to(second / 'best.pt')
                else:
                    output = paths[kind]
                self.assert_parser_error(first, second, output)
                self.assert_inputs_unchanged(originals)

    def test_verifier_refuses_dangling_symlink_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, originals = tiny_runs(root)
            destination = root / 'unrelated.json'
            output = root / 'alias.json'
            output.symlink_to(destination)
            self.assert_parser_error(first, second, output)
            self.assertTrue(output.is_symlink())
            self.assertFalse(destination.exists())
            self.assert_inputs_unchanged(originals)

    def test_verifier_refuses_output_created_during_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'raced.json'

            def competing_writer(*_):
                output.write_bytes(b'concurrent output\n')
                return {'passed': True}

            with patch.object(verify_reproduction, 'verify', side_effect=competing_writer):
                self.assert_parser_error(root / 'first', root / 'second', output)
            self.assertEqual(output.read_bytes(), b'concurrent output\n')

    def test_verifier_serializes_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'new_parent' / 'invalid.json'
            with patch.object(verify_reproduction, 'verify',
                              return_value={'passed': True, 'invalid': float('nan')}):
                self.assert_parser_error(root / 'first', root / 'second', output)
            self.assertFalse(output.exists())
            self.assertFalse(output.parent.exists())

    def test_verifier_writes_new_output_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, originals = tiny_runs(root)
            output = root / 'new_parent' / 'verification.json'
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(run_verifier(first, second, output), 0)
            saved = json.loads(output.read_text())
            self.assertEqual(saved, json.loads(stdout.getvalue()))
            self.assertTrue(saved['passed'])
            self.assertEqual(saved['tensor_count'], 1)
            self.assert_inputs_unchanged(originals)

    def test_verifier_stdout_mode_does_not_create_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, originals = tiny_runs(root)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(run_verifier(first, second), 0)
            self.assertTrue(json.loads(stdout.getvalue())['passed'])
            self.assertEqual(set(root.iterdir()), {first, second})
            self.assert_inputs_unchanged(originals)

    def test_verifier_reports_bad_output_parent_as_parser_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, originals = tiny_runs(root)
            parent = root / 'not_a_directory'
            parent.write_bytes(b'keep this file')
            self.assert_parser_error(first, second, parent / 'verification.json')
            self.assertEqual(parent.read_bytes(), b'keep this file')
            self.assert_inputs_unchanged(originals)


if __name__ == '__main__':
    unittest.main()
