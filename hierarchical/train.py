"""Train on authorized local images; keep raw data and checkpoints out of Git."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import PIL
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision.models import resnet18

from .data import load_records, split_records, fingerprint, balance_weights, ImageDataset
from .model import SoftGatedExpertModel, classification_loss


def evaluate(model, loader, device):
    model.eval()
    total = 0
    total_loss = 0.0
    gate_correct = child_correct = coherent_parent_correct = 0
    by_class = Counter()
    correct_class = Counter()
    with torch.no_grad():
        for images, parents, children in loader:
            images, parents, children = images.to(device), parents.to(device), children.to(device)
            gate, log_probs = model(images)
            loss = classification_loss(gate, log_probs, parents, children)
            predicted = log_probs.argmax(1)
            total += len(images)
            total_loss += loss.item() * len(images)
            child_correct += (predicted == children).sum().item()
            gate_correct += (gate.argmax(1) == parents).sum().item()
            coherent_parent_correct += (model.parents[predicted] == parents).sum().item()
            by_class.update(children.tolist())
            correct_class.update(children[predicted == children].tolist())
    if not total:
        raise ValueError("empty evaluation loader")
    return {'examples': total, 'loss': total_loss / total,
            'subclass_accuracy': child_correct / total,
            'subclass_macro_recall': sum(correct_class[k] / n for k, n in by_class.items()) / len(by_class),
            'gate_superclass_accuracy': gate_correct / total,
            'coherent_superclass_accuracy': coherent_parent_correct / total}


def train(args):
    if args.epochs < 1 or args.batch_size < 2 or args.learning_rate <= 0:
        raise ValueError("epochs >= 1, batch_size >= 2, learning_rate > 0 required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.json').exists() or (output / 'best.pt').exists():
        raise ValueError("output directory already contains results; choose a fresh directory")
    records, hierarchy, super_ids, sub_ids = load_records(args.data_dir)
    parts = split_records(records, seed=args.seed)
    image_root = Path(args.data_dir) / 'train_images'
    datasets = {k: ImageDataset(image_root, v, training=k == 'train') for k, v in parts.items()}
    sampler = WeightedRandomSampler(balance_weights(parts['train']), len(parts['train']),
                                    replacement=True, generator=torch.Generator().manual_seed(args.seed + 1))
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, sampler=sampler,
                             num_workers=0, drop_last=len(parts['train']) % args.batch_size == 1,
                             generator=torch.Generator().manual_seed(args.seed + 2))
    eval_loaders = {k: DataLoader(datasets[k], batch_size=args.batch_size, shuffle=False, num_workers=0)
                    for k in ('validation', 'test')}
    model = SoftGatedExpertModel(hierarchy)
    weights_hash = None
    if args.weights_path:
        path = Path(args.weights_path)
        weights_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        backbone = resnet18(weights=None)
        backbone.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        model.feature_extractor.load_state_dict(nn.Sequential(*list(backbone.children())[:-1]).state_dict())
        del backbone
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_score = -1
    best_state = None
    best_epoch = None
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for images, parents, children in train_loader:
            images, parents, children = images.to(device), parents.to(device), children.to(device)
            optimizer.zero_grad(set_to_none=True)
            gate, log_probs = model(images)
            loss = classification_loss(gate, log_probs, parents, children)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('nonfinite gradient')
            optimizer.step()
            total_loss += loss.item() * len(images)
            count += len(images)
        metrics = evaluate(model, eval_loaders['validation'], device)
        history.append({'epoch': epoch, 'training_loss': total_loss / count, 'validation': metrics})
        print(json.dumps(history[-1]), flush=True)
        if metrics['subclass_accuracy'] > best_score:
            best_score = metrics['subclass_accuracy']
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # Evaluate the test split only after selecting the checkpoint on validation.
    test_metrics = evaluate(model, eval_loaders['test'], device)
    root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((root / 'hierarchical').glob('*.py'))}
    stats = {k: {'examples': len(v), 'image_groups': len({r.pixel_hash for r in v}),
                 'sha256': fingerprint(v)} for k, v in parts.items()}
    report = {'task': 'closed-set superclass/subclass classification; no novelty-detection claim',
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'torchvision': torchvision.__version__, 'numpy': np.__version__,
                              'pillow': PIL.__version__, 'device': str(device),
                              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
              'config': {'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
                         'learning_rate': args.learning_rate, 'image_size': 64, 'hidden_dim': 256,
                         'balanced_training_only': True, 'sampler_draws_per_epoch': len(parts['train']),
                         'training_batch_singleton_dropped': len(parts['train']) % args.batch_size == 1,
                         'split_fractions': [0.6, 0.2, 0.2]},
              'pretrained_backbone_sha256': weights_hash,
              'dataset_sha256': fingerprint(records), 'source_sha256': source_hashes,
              'external_superclass_ids': super_ids, 'external_subclass_ids': sub_ids,
              'subclass_to_superclass': hierarchy, 'splits': stats,
              'pixel_group_overlap': {f'{a}/{b}': len({r.pixel_hash for r in parts[a]} & {r.pixel_hash for r in parts[b]})
                                      for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]},
              'selection': 'highest validation subclass accuracy; first epoch wins ties',
              'best_epoch': best_epoch, 'history': history, 'test': test_metrics,
              'wall_seconds': time.monotonic() - started}
    torch.save({'state_dict': best_state, 'hierarchy': hierarchy, 'superclass_ids': super_ids,
                'subclass_ids': sub_ids, 'best_epoch': best_epoch}, output / 'best.pt')
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'best_epoch': best_epoch, 'test': test_metrics}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights-path', help='Optional trusted local torchvision ResNet18 state dictionary; no automatic download')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
