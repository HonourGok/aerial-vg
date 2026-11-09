# aerial-vg — direction-aware augmentation for visual grounding in aerial images

YZM5110 Multimodal Learning, research project.

**Question.** Overhead images have no canonical "up", so vertical flips and 90° rotations are valid views of the same
scene. Referring expressions, though, contain direction words ("the ship at the *upper left*"). Does training with all
8 symmetries of the square (D4) **while rewriting the direction words consistently** improve language-guided
localization, especially for expressions with spatial words? And is any gain due to the text remapping itself, or
only to the extra geometric augmentation?

**Paper studied:** Zhan, Xiong, Yuan. *RSVG: Exploring Data and Models for Visual Grounding on Remote Sensing Data*,
IEEE TGRS 2023 ([arXiv 2210.12634](https://arxiv.org/abs/2210.12634)). The official repository
([RSVG-pytorch](https://github.com/ZhanYang-nwpu/RSVG-pytorch)) releases the dataset and a data loader but **not the
MGVLF model or training code**, so we reproduce the paper's **TransVG** baseline on DIOR-RSVG
(TransVG: [Deng et al., ICCV 2021](https://github.com/djiajunustc/TransVG); MGVLF is built on top of TransVG).

## Conditions

| name | `--geo_aug` | what it does |
|---|---|---|
| `baseline` | `hflip` | TransVG default: horizontal flip p=0.5 with left↔right swap (reproduction) |
| `d4_remap` | `d4_remap` | **proposed**: uniform random D4 element, direction words remapped |
| `d4_noremap` | `d4_noremap` | **control**: same geometry, text left unchanged |
| `no_geo` | `none` | ablation: no geometric augmentation |

Everything else is identical (seed, subset of the training split, epochs, LR schedule, crop/scale/translate
augmentation). The remapping (`datasets/geo_aug.py`) maps {left, right, top, bottom, upper, lower, leftmost, …,
above, below} through the direction permutation of each transform and keeps corner phrases in
vertical-then-horizontal order (e.g. rot90: "upper left" → "lower left"; transpose: "top-right" → "bottom-left").

## Changes to TransVG (all marked `(aerial-vg)` in the code)

- `pytorch_pretrained_bert` → Hugging Face `transformers` (`hidden_states[k]` = old `all_encoder_layers[k-1]`).
- Fixed the torchvision version check that breaks on torchvision ≥ 0.10, `collections.Iterable`, `weights=None`.
- New `datasets/dior_rsvg.py` loader (same indexing as the official RSVG loader), `--train_fraction`
  (fixed, seeded subset), `--amp`, `--max_hours` + full resume (optimizer, scheduler, scaler, best accuracy),
  per-epoch time and peak memory in `log.txt`.
- `eval_dior.py`: Pr@0.5–0.9, meanIoU, cumIoU overall and for direction-word subsets, latency, per-sample CSV.
- `compare.py`: paired bootstrap 95% CI and exact McNemar test vs. the baseline.

## Setup

```bash
pip install -r requirements.txt          # torch/torchvision: use the CUDA build for your machine
wget -P checkpoints https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
```

Data: download DIOR-RSVG from the authors' Google Drive
(<https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_>, research use only) so that
`DATA_ROOT` contains `Annotations/` and `JPEGImages/`. Split files are in `splits/` (train 26,991 / val 3,829 /
test 7,500 expressions).

## Reproduce

```bash
export DATA_ROOT=/path/to/DIOR_RSVG DETR=checkpoints/detr-r50-e632da11.pth
export EPOCHS=15 LR_DROP=10 FRACTION=0.5 BS=8 IMSIZE=640 SEED=13
python tools/dataset_stats.py --data_root $DATA_ROOT --out results/dataset_stats.json
bash scripts/run_one.sh baseline   hflip      0
bash scripts/run_one.sh d4_remap   d4_remap   0
bash scripts/run_one.sh d4_noremap d4_noremap 0
bash scripts/run_one.sh no_geo     none       0
python compare.py --runs baseline=results/baseline d4_remap=results/d4_remap \
    d4_noremap=results/d4_noremap no_geo=results/no_geo --ref baseline --out results/comparison
python tools/viz_examples.py --a results/baseline --b results/d4_remap --data_root $DATA_ROOT \
    --out results/figures/baseline_vs_d4remap.png
```

The reported runs were made on Kaggle (2× T4) with `kaggle_notebook.ipynb`, which downloads the dataset from the authors' Drive with `gdown` and runs two conditions in parallel.
A run stopped by `MAX_HOURS` resumes automatically when `run_one.sh` is called again with the same arguments.

**Small test (CPU, minutes):** `EPOCHS=1 FRACTION=0.01 bash scripts/run_one.sh _smoke hflip 0`.

**Compute note.** The paper trains 150 epochs on the full training split. Because of the course time budget we train
15 epochs on a fixed 50% subset for every condition, so absolute numbers are expected to be below the paper's;
comparisons between conditions are matched.

## Results

`results/` holds `metrics.json` and `per_sample.csv` for each condition, `comparison/table.md`, training logs,
and figures. See the report for the discussion.

Third-party code and data: see [NOTICE.md](NOTICE.md).
