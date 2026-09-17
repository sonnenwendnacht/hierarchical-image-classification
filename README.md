# Hierarchical image classification

[![Tests](https://github.com/sonnenwendnacht/hierarchical-image-classification/actions/workflows/tests.yml/badge.svg)](https://github.com/sonnenwendnacht/hierarchical-image-classification/actions/workflows/tests.yml)

A hierarchical image classifier combining a ResNet18 backbone, a superclass
gate, and per-superclass subclass experts. This code-only repository presents
**Junzhe Zong's model2 contribution to a team COMS 4776 project**, developed
with Matt and Adarsh Pachori.
[Team attribution and source history](PROVENANCE.md).

For a subclass `c` with parent superclass `p`, the model computes:

```text
P(c | image) = P(p | image) × P(c | p, image)
```

Original notebook sources are preserved separately. September 2026 AI-assisted
maintenance adds a tested extraction with corrected data splitting. Evaluation
is limited to known classes; the original novelty heuristics are not validated
as an open-set recognition system.

[Offline demo](#quick-start-no-dataset-or-model-download) ·
[Recorded evaluation](#recorded-held-out-evaluation) ·
[Local training](#training-on-authorized-local-data) · [Team provenance](PROVENANCE.md)

The recorded five-epoch experiment, initialized with ImageNet ResNet18 weights,
achieved 79.84% subclass accuracy on a 1,260-image held-out split after
validation-only checkpoint selection. This is one fixed-seed, known-class
experiment with exact-pixel grouping, not a
cross-seed or near-duplicate-free benchmark. The course dataset is not bundled.

## Quick start: no dataset or model download

From the repository root, use Python 3.12 and the CPU dependencies below:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m hierarchical.demo
```

The demo uses random image tensors and randomly initialized weights to verify
forward/backward execution; its output is not an accuracy result. Tests include
equivalence to the original notebook model under matching parameters,
probability normalization, finite gradients, hierarchy-consistent predictions,
exact-pixel-group splitting, deterministic evaluation transforms, and a complete
small CPU training/checkpoint round trip.

All 25 tests passed in both a Python 3.12 CPU environment and the existing
Python 3.14 environment. Hosted CI repeats the CPU installation and tests.

## Recorded held-out evaluation

The recorded experiment uses 6,288 local course images: 87 observed subclasses
under bird, dog, and reptile superclasses. The data's upstream redistribution
terms are unverified, so no course images or annotations are uploaded.

| Split / metric | Result |
| --- | ---: |
| Training / validation / test images | 3,768 / 1,260 / 1,260 |
| Exact decoded-image overlap between splits | 0 |
| Training budget | 5 epochs |
| Selected epoch, using validation accuracy | 5 |
| Held-out subclass accuracy | **79.84%** |
| Held-out macro subclass recall | 79.83% |
| Held-out superclass accuracy, consistent label pairs | 99.60% |

See [the machine-readable record](evaluation/seed42-five-epochs.json) for every
epoch, configuration, data/split/code hashes, pretrained-weight hash, and runtime
versions. The test split was evaluated only after checkpoint selection on the
validation split. Balancing and random augmentation apply only to training.

This result and its reproduction record are preserved from release
`f5850ea4e5913cbc2a64a718d14053665dd17c0f`, before the later
[numerical guards](#numerical-and-small-dataset-guards).
Their source hashes identify that release, not the newer guard code. The full
course-image experiment was not rerun for this follow-up, and its scores were
not overwritten. Use the recorded revision for exact historical reproduction;
the verification script intentionally reports a source mismatch on newer code.

A second full run with the same configuration reproduced every metric and all
160 checkpoint tensors exactly. See the [reproduction check](evaluation/reproduction.json).
This is same-seed reproducibility, not a second independent statistical trial.

```sh
python verify_reproduction.py runs/first-run runs/second-run
```

This is one fixed-seed, short training experiment—not an estimate across
independent seeds, a comparison with other architectures, or a leaderboard
result. Exact-pixel grouping does not detect near-duplicates. The preserved
notebooks split an already balanced dataset and use a different training budget,
so their saved scores are not directly comparable.

## Numerical and small-dataset guards

The September 16 follow-up rejects nonfinite model outputs or evaluation loss
before computing predictions or selecting a checkpoint. Learning rates must be
finite and positive, and the final training split must contain at least two
examples for the batch-normalized heads. JSON serialization is strict and is
validated before either checkpoint or metrics is written.

Six new regression tests cover these failure cases and hand-calculated metrics
with unequal class frequencies and a partial final batch. A separate one-epoch
run on 24 generated images matched the pre-fix implementation's metrics, all
150 checkpoint tensors, and Python/NumPy/PyTorch RNG states exactly. This checks
that valid training behavior is preserved; it is not a real-data accuracy run.

## Training on authorized local data

Expected layout (not distributed):

```text
your-data/
  train_data.csv   # image, superclass_index, subclass_index
  train_images/    # filenames referenced by the CSV
```

Labels are mapped from observed training-table IDs. Each subclass must have a
single parent and at least three distinct image groups so it can appear in all
three splits. The final training split must also contain at least two images;
three total image groups in a one-subclass dataset are insufficient for training.
Additional annotation columns are ignored.

```sh
python -m hierarchical.train \
  --data-dir /path/to/your-data \
  --output runs/my-experiment \
  --epochs 5 --seed 42 --device cpu
```

This command starts from random weights. The recorded experiment instead used
an existing local torchvision ImageNet-1K ResNet18 state dictionary, passed with
`--weights-path /path/to/resnet18-f37072fd.pth`, and `--device cuda` in the
pre-existing GPU environment documented in the record. Only load a trusted
checkpoint. There is no automatic data or weight download, and a run refuses to
overwrite an existing metrics/checkpoint result.

The recorded GPU environment is a pre-existing PyTorch nightly installation;
the pinned CPU environment is for installation, model equivalence, and pipeline
tests. Identical numerical results across those environments are not promised.

## Scope and limits

- Known-class classification only. The unrepresented `novel` mapping entries are
  not learned classes; original threshold heuristics are retained only in the
  historical notebooks. No novelty-detection accuracy or threshold calibration
  is claimed.
- No raw datasets, prediction CSVs, image outputs, pretrained weights, or trained
  checkpoints are committed. Full real-data reproduction requires authorized
  access to the original data and the recorded pretrained weights.
- The backbone and gated-expert idea are established techniques. What this
  repository demonstrates is the team's application, Junzhe's model2 work, and
  explicitly documented maintenance and verification.
- The private [team repository](https://github.com/MatthewLee72/coms4776-nndl-final-project)
  is unchanged. This public copy intentionally excludes its private history and
  other teammates' model implementations.
