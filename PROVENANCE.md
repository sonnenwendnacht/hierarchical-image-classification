# Source and contribution record

## Team work, not sole authorship

This portfolio copy focuses on Junzhe Zong's `model2.ipynb` contribution to the
COMS 4776 Neural Networks and Deep Learning team project. The source repository
is [MatthewLee72/coms4776-nndl-final-project](https://github.com/MatthewLee72/coms4776-nndl-final-project)
(private/access restricted at preparation time).

The team repository contains work by Junzhe Zong (`sonnenwendnacht`), Matt
(`MatthewLee72`), and Adarsh Pachori. Matt and Adarsh developed additional models
and shared project material. This code-only copy is **not** a complete mirror,
does not replace the team's repository, and does not claim those contributions
as Junzhe's individual work.

Relevant source history:

| Commit | Recorded author | Model2 change |
| --- | --- | --- |
| `3b13f1c29ab5667b58ca56dd288c542ce0558a49` | sonnenwendnacht | Initial model2 notebook |
| `62e0e99d` | sonnenwendnacht | Hard-logit inference update |
| `35666fca` | sonnenwendnacht | Hard-logit threshold update |
| `6cabb876` | Junzhe | Pre-meeting version retained on PC |
| `5c1d2d06` | Matt | Data augmentation changes to both models |

The transferred laptop checkout is at team commit
`42e284e1a22a70a39365c04046fe49cbfab0e667`, with additional uncommitted changes in
`model2.ipynb`. The working laptop notebook is not represented as an already
committed or reviewed team release. Author labels in Git establish recorded
contributions, not whether a particular line was written with AI assistance.

## Preserved material

- `historical/model2.pc.ipynb`: PC notebook source cells.
- `historical/model2.laptop.ipynb`: newer laptop working notebook source cells.
- `historical/model2_class.py`: exact laptop model-definition cell; its `torch`,
  `nn`, and `F` globals originally came from earlier notebook cells.
- `historical/manifest.json`: original/export SHA-256 hashes and transformation
  record. Notebook cell sources are unchanged; outputs, execution counts,
  attachments, and notebook metadata were removed. These exports are therefore
  not byte-identical copies of the full notebooks.

The original PC and laptop notebooks, source histories, and data were not
modified. Raw images, annotation CSVs, descriptions, prediction CSVs,
checkpoints, workspace settings, and the private repository's Git history are
not included here. A fresh public history avoids exposing excluded material
through older commits. The owner confirmed publication clearance in September
2026; this does not establish a redistribution license for the course dataset.

## September 2026 maintenance, with Codex assistance

The maintained model preserves the ResNet18 backbone, batch-normalized gate and
expert heads, and the superclass-times-conditional-subclass factorization.
An automated test loads identical parameters into the unchanged notebook class
and maintained model, then checks matching outputs for the same known-class
hierarchy. The maintained model computes log probabilities directly for numeric
stability and registers hierarchy tensors as model buffers.

The following are new maintenance changes, not original research claims:

- Split original image groups into training/validation/test sets before any
  sampling or augmentation. Exact decoded-pixel duplicates stay in one split.
- Balance only training draws; evaluation uses every held-out image once with
  deterministic preprocessing. Missing/corrupt files fail rather than becoming
  black images.
- Build gate/experts only for observed classes. The local data has 3 represented
  superclasses and 87 represented subclasses. Its mapping files also have
  `novel` entries, but no training examples for those labels.
- Return hierarchy-consistent label pairs from `predict()` by deriving the
  superclass from the most probable joint subclass. Gate-only accuracy is
  reported separately.
- Add tests, an offline synthetic-tensor demo, a standalone training entry
  point, pinned CPU dependencies, CI, and new five-epoch evaluation records.
- September 16 numerical follow-up: reject nonfinite evaluation outputs/loss,
  nonfinite learning rates, and a one-example training split; validate strict
  JSON before output artifacts are written. These are failure-boundary guards,
  not model changes. Six added regression methods and a synthetic before/after
  training comparison preserve the valid path. Earlier real-data evaluation
  records and all historical sources remain unchanged at their recorded hashes.
- Output-safety follow-up: reserve a fresh training directory atomically before
  data loading, retain failed runs, and write verification records exclusively
  without replacing existing files or input aliases. Ten added tests exercise
  reservation, output races, aliases, strict serialization, and valid CLI use;
  model computation, evaluation, and recorded artifacts are unchanged.

The original balanced-dataset split can repeat a source image across training
and validation. Its probability thresholds, hard maximum-logit thresholds, and
saved scores were not independently validated for novelty detection. The
historical code also uses `len(sub_map_df)` as a novel subclass ID even though
the mapping file already includes a novel row. These issues are why this release
reports a fresh **closed-set** experiment and no open-set/leaderboard claim.

## External foundations

- [ResNet paper](https://arxiv.org/abs/1512.03385): backbone architecture by He et al.
- [Torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html): implementation and optional ImageNet-1K weights.

ResNet, ImageNet pretraining, and gated-expert modeling are not claimed as novel
contributions of this project.
