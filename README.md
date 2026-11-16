# aerial-vg — direction-consistent D4 augmentation for visual grounding in aerial images

YZM5110 Multimodal Learning, research project — Onur Gökdoğan.
Report: `Gokdogan_YZM5110_Research.pdf`. Submitted version: tag **`v1.0`**.

**Question.** Overhead images have no canonical "up", so flips and 90° rotations are valid views of the same scene.
Referring expressions, though, contain direction words ("the ship at the *upper left*"). Does training with all
8 symmetries of the square (D4) **while rewriting the direction words consistently** improve grounding? And is any
gain due to the text remapping itself, or only to the extra geometric augmentation?

**Reproduction target.** TransVG ([Deng et al., ICCV 2021](https://github.com/djiajunustc/TransVG)) on DIOR-RSVG, as
reported by Zhan, Xiong, Yuan, *RSVG*, IEEE TGRS 2023 ([arXiv 2210.12634](https://arxiv.org/abs/2210.12634)):
72.41 Pr@0.5. The RSVG repository releases the dataset and a loader but not its training code, so TransVG is
re-implemented here.

## Results (test, 7,500 expressions, seed 13, 7 epochs, one T4 per run)

| run | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
|---|---|---|---|---|---|---|---|
| **d4_remap** (proposed) | **48.33** | **41.13** | **31.68** | **17.79** | **3.23** | **42.03** | **61.18** |
| baseline (TransVG recipe) | 46.44 | 39.33 | 30.28 | 17.25 | 3.12 | 40.96 | 58.94 |
| no_geo (ablation) | 45.61 | 38.48 | 28.97 | 15.59 | 3.19 | 40.26 | 58.18 |
| d4_noremap (control) | 28.23 | 23.92 | 17.83 | 9.93 | 1.68 | 26.55 | 43.33 |

d4_remap − baseline: +1.89 Pr@0.5, paired bootstrap 95% CI [+1.15, +2.61], exact McNemar p = 2.3e-7
(`results/comparison/table.md`). The CI covers test-set sampling only; with one training seed per condition the
gain is not yet separated from seed-to-seed variation (seeds 14 and 15: see *Extra seeds*). Absolute numbers are far
below the published 72.41 because of the short schedule (7 vs. 150 epochs) and the deviations listed below.

## Conditions

| name | `--geo_aug` | what it does |
|---|---|---|
| `baseline` | `hflip` | TransVG default: horizontal flip p=0.5 with left↔right swap (reproduction) |
| `d4_remap` | `d4_remap` | **proposed**: uniform random D4 element, direction words remapped |
| `d4_noremap` | `d4_noremap` | **control**: same geometry, text left unchanged |
| `no_geo` | `none` | ablation: no geometric augmentation |

Everything else is identical: seed 13, full training split, 7 epochs, LR ×0.1 after epoch 5, batch 8, fp16 AMP,
crop/scale/translate and colour augmentation, 640×640 input, 40 tokens, checkpoint selected on the first 500
validation expressions, test set evaluated once. The remapping (`datasets/geo_aug.py`) maps {left, right, top,
bottom, upper, lower, leftmost, …, above, below} through the direction permutation of each transform and keeps corner
phrases in vertical-then-horizontal order (rot90: "upper left" → "lower left"; transpose: "top-right" → "bottom-left").

## Changes to TransVG (all marked `(aerial-vg)` in the code)

- **Training fix (applies to every condition).** In this environment the original post-LayerNorm fusion
  transformer collapsed to one constant box for every input (loss stuck at ~1.85). The fusion block is now pre-LN
  (`--vl_prenorm 1`), with a 500-iteration linear LR warm-up (`--warmup_iters 500`) and gradient clipping 0.1
  (`--clip_max_norm 0.1`); `scripts/run_one.sh` passes all three. The Kaggle Version 5 run applied the same patch
  from a notebook cell (see `notebooks/aerial-vg-experiments_v5_as_run.ipynb`, cell 4) to the files whose MD5 is in
  `results/results_commit.txt`; tag `v1.0` contains that patch in the code itself, with no other change to model or
  training code.
- `pytorch_pretrained_bert` → Hugging Face `transformers` (`hidden_states[k]` = old `all_encoder_layers[k-1]`).
- Fixed the torchvision version check that breaks on torchvision ≥ 0.10, `collections.Iterable`, `weights=None`.
- New `datasets/dior_rsvg.py` loader (same indexing as the official RSVG loader), `--train_fraction`, `--amp`,
  `--max_hours` + full resume, per-epoch time and peak memory in `log.txt`.
- Box loss is L1 + GIoU as in the public TransVG code (the paper says smooth-L1).

## Setup

```bash
git clone https://github.com/HonourGok/aerial-vg && cd aerial-vg && git checkout v1.0
pip install torch==2.11.0 torchvision==0.26.0      # CUDA build for your machine; on Kaggle keep the preinstalled one
pip install -r requirements.txt
wget -P checkpoints https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
```

`bert-base-uncased` is downloaded from the Hugging Face Hub on first use.

**Data (licensed, not redistributed here).** DIOR-RSVG is released by the RSVG authors for research use only
(CC BY-NC 4.0) in their Google Drive folder
<https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_> (`Annotations.zip`, `JPEGImages.zip`,
~5 GB). Unzip so that `DATA_ROOT` contains `Annotations/` and `JPEGImages/`. The split files in `splits/` are the
released ones (train 26,991 / val 3,829 / test 7,500 expressions). The Kaggle notebook downloads the data with
`gdown` and prints the MD5 of both zips.

## Reproduce the reported runs

```bash
export DATA_ROOT=/path/to/DIOR_RSVG DETR=checkpoints/detr-r50-e632da11.pth
python tools/preresize.py --src $DATA_ROOT --dst ${DATA_ROOT}_640 --size 640   # resize once to the model input size
export DATA_ROOT=${DATA_ROOT}_640
export EPOCHS=7 LR_DROP=5 FRACTION=1.0 BS=8 IMSIZE=640 SEED=13 VAL_MAX=500 MAX_HOURS=5
python tools/dataset_stats.py --data_root $DATA_ROOT --split_root splits --out results/dataset_stats.json
bash scripts/run_one.sh baseline   hflip      0      # ~3.5 h on one T4 (train + test)
bash scripts/run_one.sh d4_remap   d4_remap   0
bash scripts/run_one.sh d4_noremap d4_noremap 0
bash scripts/run_one.sh no_geo     none       0
```

The reported runs used `kaggle_notebook.ipynb` (`MODE = 'main'`) on Kaggle with 2× T4, two conditions in parallel.
A run stopped by `MAX_HOURS` resumes when `run_one.sh` is called again with the same arguments.

## Reproduce the tables and figures from the saved outputs (no GPU, no data)

```bash
python compare.py --runs baseline=results/baseline d4_remap=results/d4_remap \
    d4_noremap=results/d4_noremap no_geo=results/no_geo --ref baseline --out results/comparison   # main table, CIs, McNemar
python tools/analysis.py --results results --out results/analysis   # size groups, stated-side test, disagreements, val curves
python tools/seed_summary.py --results results                        # mean ± sd over seeds (once seed runs exist)
```

`tools/viz_examples.py` draws `results/figures/baseline_vs_d4remap.png` (needs the images).

| Report item | Produced by |
|---|---|
| Table in §7 (main result), subset chart, uncertainty table | `results/<run>/metrics.json`, `compare.py` → `results/comparison/` |
| Size groups, class mentions, "why remapping matters", unfavourable/favourable counts, training curves | `tools/analysis.py` → `results/analysis/` |
| Direction-word shares (§1, §6) | `tools/dataset_stats.py` → `results/dataset_stats.json` |
| Training time, memory, latency | `results/<run>/train_log.txt`, `metrics.json` |
| Example figure | `tools/viz_examples.py` → `results/figures/` |

## Small test

```bash
python tests/test_geo_aug.py          # CPU, seconds, needs only torch + numpy + Pillow
```

It checks for all 8 D4 elements that the transformed box matches where the pixels move, that a remapped direction
word still names the object's side, and that `hflip` / `d4_noremap` / `none` behave as described. An end-to-end
smoke run (needs the data): `DEVICE=cpu EPOCHS=1 FRACTION=0.01 VAL_MAX=40 TEST_LIMIT=200 LAT=5 bash scripts/run_one.sh _smoke hflip 0`.

## Extra seeds

`kaggle_notebook.ipynb` with `MODE = 'seeds'` trains `baseline` and `d4_remap` with seeds 14 and 15 (~7.3 h on 2× T4)
into `results/<name>_s14`, `_s15`, runs `tools/seed_summary.py`, and checks image overlap between the splits
(`results/split_image_overlap.json`).

## Orientation-robustness test (optional, not in the report)

`tools/eval_rotations.py` evaluates trained models on test images rotated or flipped by one fixed transform, with the
expressions remapped (`--no_remap` keeps the original text).

## Repository layout

```
train.py engine.py eval_dior.py compare.py     training / evaluation / paired comparison
models/ utils/ datasets/                       TransVG (adapted) + DIOR-RSVG loader + geo_aug.py (D4 remapping)
scripts/run_one.sh scripts/launch.py           one condition / two conditions in parallel
tools/                                         preresize, dataset_stats, analysis, seed_summary, viz, benchmark
tests/test_geo_aug.py                          small CPU test
kaggle_notebook.ipynb                          notebook for the Kaggle runs (MODE = 'main' | 'seeds')
notebooks/aerial-vg-experiments_v5_as_run.ipynb   the executed Version 5 notebook behind the reported numbers
results/                                       metrics, per-sample predictions, logs, comparison, analysis, figure
splits/                                        released DIOR-RSVG split files
```

No credentials, dataset images or model checkpoints are stored here (`checkpoints/`, `outputs/` are git-ignored).

## AI use

Claude (Anthropic) assisted with this project: diagnosing the fusion-transformer collapse and writing the fix,
running the Kaggle notebooks, porting the code and writing the evaluation and analysis tools, extracting facts from
papers, and drafting the report. The research question, the experimental design, the D4 remapping module and the
selection and reading of the literature are the author's own. Details are in the report's AI-use statement.

Third-party code and data: see [NOTICE.md](NOTICE.md).
