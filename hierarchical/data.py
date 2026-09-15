"""Local-only image loading and image-group-disjoint splits before balancing."""
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class Record:
    image: str
    superclass: int
    subclass: int
    pixel_hash: str


def load_records(data_dir):
    root = Path(data_dir).resolve()
    image_root = (root / "train_images").resolve()
    with (root / "train_data.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image", "superclass_index", "subclass_index"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("annotation CSV must have image, superclass_index, subclass_index")
        rows = list(reader)
    if not rows:
        raise ValueError("empty annotation table")
    super_ids = sorted({int(r['superclass_index']) for r in rows})
    sub_ids = sorted({int(r['subclass_index']) for r in rows})
    if min(super_ids + sub_ids) < 0:
        raise ValueError("label IDs must be nonnegative")
    super_index = {v: i for i, v in enumerate(super_ids)}
    sub_index = {v: i for i, v in enumerate(sub_ids)}
    parents = {}
    records = []
    seen_names = set()
    for row in rows:
        name = row['image']
        if not name or '/' in name or '\\' in name or Path(name).name != name:
            raise ValueError("image names must be plain filenames")
        if name in seen_names:
            raise ValueError("duplicate image name in annotations")
        seen_names.add(name)
        path = (image_root / name).resolve()
        if path.parent != image_root:
            raise ValueError("image path escapes train_images")
        super_id = super_index[int(row['superclass_index'])]
        sub_id = sub_index[int(row['subclass_index'])]
        if sub_id in parents and parents[sub_id] != super_id:
            raise ValueError("a subclass maps to multiple superclasses")
        parents[sub_id] = super_id
        # Fail on missing/corrupt images instead of silently substituting black.
        with Image.open(path) as image:
            image = image.convert('RGB')
            digest = hashlib.sha256(str(image.size).encode() + b'\0' + image.tobytes()).hexdigest()
        records.append(Record(name, super_id, sub_id, digest))
    return records, [parents[i] for i in range(len(sub_ids))], super_ids, sub_ids


def split_records(records, seed=42, val_fraction=0.2, test_fraction=0.2):
    """Stratify image groups per subclass; exact decoded duplicates stay together."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be positive and sum to less than one")
    groups = defaultdict(list)
    for row in records:
        groups[row.pixel_hash].append(row)
    by_class = defaultdict(list)
    for key, group in groups.items():
        if len({(r.superclass, r.subclass) for r in group}) != 1:
            raise ValueError("identical image pixels have conflicting labels")
        by_class[group[0].subclass].append(key)
    result = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for label in sorted(by_class):
        keys = sorted(by_class[label])
        if len(keys) < 3:
            raise ValueError("each subclass needs at least three distinct image groups")
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * val_fraction))
        n_test = max(1, round(len(keys) * test_fraction))
        if n_val + n_test >= len(keys):
            raise ValueError("requested fractions leave no training group for a subclass")
        partitions = {"validation": keys[:n_val], "test": keys[n_val:n_val + n_test],
                      "train": keys[n_val + n_test:]}
        for name, selected in partitions.items():
            for key in selected:
                result[name].extend(sorted(groups[key], key=lambda r: r.image))
    return result


def fingerprint(records):
    rows = sorted((r.image, r.superclass, r.subclass, r.pixel_hash) for r in records)
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def balance_weights(records):
    counts = Counter(r.subclass for r in records)
    return [1.0 / counts[r.subclass] for r in records]


def image_transform(training=False):
    operations = []
    if training:
        operations += [transforms.RandomRotation(15),
                       transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                       transforms.RandomHorizontalFlip()]
    operations += [transforms.Resize((64, 64)), transforms.ToTensor(),
                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(operations)


class ImageDataset(Dataset):
    def __init__(self, image_root, records, training=False):
        self.root = Path(image_root).resolve()
        self.records = list(records)
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        path = (self.root / row.image).resolve()
        if path.parent != self.root:
            raise ValueError("image path escapes dataset root")
        with Image.open(path) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, row.superclass, row.subclass
