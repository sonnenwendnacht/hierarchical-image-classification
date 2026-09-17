"""Compare two local training runs, without uploading their checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def verify(first, second):
    reports = [json.loads((p / 'metrics.json').read_text()) for p in [first, second]]
    checkpoints = [torch.load(p / 'best.pt', weights_only=True, map_location='cpu') for p in [first, second]]
    a, b = checkpoints
    keys_equal = a['state_dict'].keys() == b['state_dict'].keys()
    tensors_equal = keys_equal and all(torch.equal(v, b['state_dict'][k]) for k, v in a['state_dict'].items())
    metadata_equal = {k: v for k, v in a.items() if k != 'state_dict'} == {k: v for k, v in b.items() if k != 'state_dict'}
    metrics_equal = {k: v for k, v in reports[0].items() if k != 'wall_seconds'} == {k: v for k, v in reports[1].items() if k != 'wall_seconds'}
    root = Path(__file__).resolve().parent
    source_matches = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                         for name, digest in reports[0]['source_sha256'].items())
    return {'scope': 'same configuration, seed, environment and data; not an independent-seed experiment',
            'report_sha256': [hashlib.sha256((p / 'metrics.json').read_bytes()).hexdigest() for p in [first, second]],
            'tensor_count': len(a['state_dict']), 'all_tensors_identical': tensors_equal,
            'checkpoint_metadata_identical': metadata_equal,
            'reports_identical_except_wall_seconds': metrics_equal,
            'recorded_source_hashes_match_current_code': source_matches,
            'passed': bool(tensors_equal and metadata_equal and metrics_equal and source_matches)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--output', type=Path, help='write a new JSON file; existing paths are refused')
    args = parser.parse_args()
    try:
        if args.output and (args.output.exists() or args.output.is_symlink()):
            raise ValueError("output already exists; choose a new file")
        result = verify(args.first, args.second)
        rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also protects against races and input aliases.
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(rendered)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(rendered, end='')
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
