# GeoRel-CLIP

**GeoRel-CLIP** is a geometry-aware relational post-pre-training method for vision-language alignment, built on top of [CLIP-Refine](https://github.com/nttcslab/clip-refine) (CVPR 2025). It extends the baseline with **AnchorBank**, **anchor-diversity regularization (Ldiv)**, **uniformity regularization (Luni)**, and **Relational KD (RKD)**.

- **Proposed method**: GeoRel-CLIP — `config/01_post-pre-training/GeoRel-CLIP.yaml`
- **Baseline**: CLIP-Refine — `config/01_post-pre-training/clip-refine.yaml`
- **Default backbone**: CLIP ViT-B/32

## Method Overview

| Component | Description |
|-----------|-------------|
| InfoNCE | CLIP contrastive loss (`lambda_cont`) |
| AnchorBank | K-means initialized anchors with periodic rotation |
| Ldiv | KL divergence on anchor usage distribution |
| Luni | Feature uniformity regularization |
| RKD | Frozen-teacher relational similarity-matrix distillation |

During training, contrastive `val_loss` on the COCO validation split is used to select the best checkpoint. Zero-shot classification accuracy on Oxford Pets is reported each epoch for monitoring.

## Requirements

| Item | Version |
|------|---------|
| Python | >= 3.10 |
| CUDA | >= 12.1 (12.3+ recommended) |
| GPU memory | >= 24 GB (batchsize=512, ViT-B/32) |

### Quick Install

```bash
cd /root/autodl-tmp
bash env/setup.sh
```

Or install manually:

```bash
pip install -r requirements.txt
```

Training scripts automatically fix invalid OpenMP thread settings (e.g. `OMP_NUM_THREADS=0`) before importing PyTorch. See `util/fix_env.py`.

### Apptainer / Singularity (optional)

```bash
apptainer build georel-clip.sif apptainer/config.def
```

## Data Preparation

> COCO paths in `data/coco_caption.py` are resolved relative to **`/root`**. Launch training from `/root`.

### 1. COCO Caption 2017 (post-pre-training)

```text
/root/autodl-tmp/dataset/coco/
├── train2017/
└── annotations/
    └── captions_train2017.json
```

Download from the [COCO website](https://cocodataset.org/#download).

### 2. Oxford Pets (in-training classification eval)

```text
/root/autodl-tmp/dataset/OxfordPets/
├── images/
└── annotations/
    ├── trainval.txt
    └── test.txt
```

Class names: `data/classnames/oxford_pets.txt`

### 3. Anchor feature file (required for GeoRel-CLIP)

```text
/root/autodl-tmp/anchor_feats_coco.pt
```

Must contain key `joint_feats` with shape `[N, D]`. Training fails at startup if `use_anchor_bank: true` and this file is missing.

### 4. Full zero-shot benchmark (optional)

`main/test.py` evaluates on multiple datasets defined in `main/experimental_settings.py`. Place each dataset under `dataset/` as specified in `data/generic.py`.

## Training

### GeoRel-CLIP (proposed)

```bash
cd /root

python autodl-tmp/main/train.py \
  --config_path autodl-tmp/config/01_post-pre-training/GeoRel-CLIP.yaml \
  --results_dir ./result \
  --experiment_id 0 \
  --num_worker 16 \
  --seed 42
```

### CLIP-Refine baseline

```bash
cd /root

python autodl-tmp/main/train.py \
  --config_path autodl-tmp/config/01_post-pre-training/clip-refine.yaml \
  --results_dir ./result \
  --experiment_id 0
```

### Common CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config_path` | — | Path to YAML config |
| `--results_dir` | `./result` | Output root directory |
| `--experiment_id` | `0` | Experiment index |
| `--resume` | `""` | Checkpoint path for resume |
| `--evaluate` | off | Run validation only |

Example output layout:

```text
result/GeoRel-CLIP-CLIP-ViT-B_32-COCOCaption/experiment0/
├── GeoRel-CLIP.yaml
├── log
└── best_model_*.pt
```

## Evaluation

Zero-shot classification using the best checkpoint in an experiment folder:

```bash
cd /root

python autodl-tmp/main/test.py \
  --config_path autodl-tmp/config/01_post-pre-training/GeoRel-CLIP.yaml \
  --results_dir ./result \
  --experiment_id 0
```

## Project Layout

```text
autodl-tmp/
├── main/                    # train.py, test.py
├── updater/georel_clip.py   # GeoRel-CLIP trainer (GeoRelCLIPUpdater)
├── model/mod_clip.py        # CLIP wrapper
├── anchor_bank_kmeans.py    # AnchorBank
├── data/                    # Dataset definitions
├── config/                  # Experiment configs
├── evaluator/               # Contrastive & classification evaluators
├── metric/                  # Zero-shot benchmark scripts
├── loss/contrastive.py      # Contrastive loss (validation)
├── util/                    # Utilities
├── env/setup.sh             # Environment setup
└── requirements.txt
```

## Configurations

| File | Role |
|------|------|
| `config/01_post-pre-training/GeoRel-CLIP.yaml` | **GeoRel-CLIP** (proposed) |
| `config/01_post-pre-training/clip-refine.yaml` | **CLIP-Refine** baseline |
| `config/01_post-pre-training/contrastive.yaml` | Contrastive-only baseline |

## Citation

If you use CLIP-Refine as the baseline, please cite:

```bibtex
@inproceedings{Yamaguchi_CVPR25_CLIP-Refine,
  title={Post-pre-training for Modality Alignment in Vision-Language Foundation Models},
  author={Yamaguchi, Shin'ya and Feng, Dewei and Kanai, Sekitoshi and Adachi, Kazuki and Chijiwa, Daiki},
  booktitle={CVPR},
  year={2025}
}
```

## License

See [LICENSE.txt](LICENSE.txt).
