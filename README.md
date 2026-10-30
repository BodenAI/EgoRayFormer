<p align="center">
  <img src="docs/logo/en_logo.png" alt="BODENAI" width="220">
</p>

<h1 align="center">EgoRayFormer</h1>

<p align="center">English | <a href="README_zh.md">中文</a></p>

<p align="center"><b>EgoRayFormer: Ray-Space Stereo Refinement for Egocentric 3D Human Pose Estimation</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.26641"><img src="https://img.shields.io/badge/arXiv-2610.26641-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

<p align="center">Haozhe Wang, Songyuan Wu, Jihai Xu, Jie Zhao · Boden AI</p>

Official implementation of [EgoRayFormer](https://arxiv.org/abs/2610.26641), a causal 3D human pose estimator for
head-mounted stereo fisheye cameras. For each joint, the decoder projects shared 3D hypotheses along and around its
viewing ray into both images. A geometric branch then corrects the projected joint locations and intersects the
resulting rays, and a learned gate blends this geometric estimate with a learned pose update.

![EgoRayFormer architecture](assets/fig_architecture.png)

## Results

Device-relative protocol on the official test splits, mm, mean ± s.d. over three seeds (Table 1 of the paper). The
encoder is DINOv3 ViT-S+/16 at 384 × 384. Every sequence is processed as one causal stream in FP32.

| Training data | Test set | MPJPE | PA-MPJPE |
|---|---|---|---|
| UnrealEgo2 | UnrealEgo2 | 17.64 ± 0.12 | 16.86 ± 0.08 |
| UnrealEgo-RW | UnrealEgo-RW | 49.55 ± 0.40 | 40.59 ± 0.30 |
| UnrealEgo2, then UnrealEgo-RW | UnrealEgo-RW | 47.57 ± 0.34 | 37.21 ± 0.46 |

See the paper for the comparison with other methods.

## Installation

Tested with Python 3.10, PyTorch 2.4.0 (CUDA 12.1), PyTorch Lightning 2.4.0, timm 1.0.29 and NumPy 1.26 on NVIDIA H200
GPUs. No compiled CUDA operators are needed.

```bash
git clone https://github.com/BodenAI/EgoRayFormer.git
cd EgoRayFormer
conda create -n egorayformer python=3.10 -y
conda activate egorayformer
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

Run all commands from the repository root. `run.py` and the scripts add it to `sys.path`, so the editable install is
optional.

**Pretrained backbones.**
* ResNet-18: torchvision ImageNet weights, downloaded on first use.
* DINOv3 ViT-S+/16: timm `vit_small_plus_patch16_dinov3.lvd1689m`. These weights are gated on Hugging Face: accept
  the DINOv3 license and set `HF_TOKEN`, or put `<timm name>.safetensors` into a directory and point
  `EGORAYFORMER_TIMM_WEIGHTS_DIR` to it.
* Without a stage-1 checkpoint, training stops if the requested pretrained weights cannot be loaded, instead of
  continuing from other weights. `EGORAYFORMER_ALLOW_WEIGHT_FALLBACK=1` allows the fallback.

## Pretrained models

The checkpoints of Table 1 will be released soon.

| Model | Training data | Test set | Checkpoint |
|---|---|---|---|
| EgoRayFormer (ViT-S+/16, 384²) | UnrealEgo2 | UnrealEgo2 | Coming soon |
| EgoRayFormer (ViT-S+/16, 384²) | UnrealEgo-RW | UnrealEgo-RW | Coming soon |
| EgoRayFormer (ViT-S+/16, 384²) | UnrealEgo2, then UnrealEgo-RW | UnrealEgo-RW | Coming soon |

A checkpoint is evaluated with the configs it was trained with (see [Training](#training)), e.g. the UnrealEgo-RW model
on the validation split:

```bash
python run.py validate --config configs/base_unrealego.yaml --config configs/data_device_relative.yaml \
  --config configs/res_256.yaml --config configs/data_rw_only.yaml --config configs/egorayformer_full.yaml \
  --config configs/ablations/no_p4.yaml --config configs/enc_sfvit_s.yaml --config configs/invalid_view_mask.yaml \
  --config configs/res_384.yaml --model.model.encoder.pretrained false --data.root /data/unrealego/processed \
  --trainer.devices 1 --trainer.strategy auto --model.eval_precision fp32 --data.eval_clip_len 0 \
  --ckpt_path <checkpoint.ckpt>
```

## Data

The experiments use [UnrealEgo2 and UnrealEgo-RW](https://unrealego.mpi-inf.mpg.de/); their terms of use are set by
the dataset authors. The staging scripts download the official files, convert them to the unified record format of
`egorayformer/data/records.py` and delete the raw data:

```bash
export DATA_ROOT=/path/to/unrealego                 # raw downloads and processed records (default /data/unrealego)
bash scripts/fetch_unrealego_meta.sh                # official split files and calibrations (small)
bash scripts/stage_unrealego_rw.sh                  # UnrealEgo-RW, ~180 GB raw -> 512^2 records + per-session rigs
bash scripts/stage_unrealego2_stream.sh train_val   # UnrealEgo2, ~3 TB raw, one sequence at a time -> 384^2 records
bash scripts/stage_unrealego2_stream.sh test
```

Records are written to `$DATA_ROOT/processed/<dataset>/<split>/`. The rig of every UnrealEgo-RW capture session is fitted
to the training annotations only. The staging scripts need Linux, `curl`, `unzip`, `tar` and `flock`, and take their
options from environment variables documented at the top of each script. Use `http_proxy` / `https_proxy` if you need a
proxy.

## Training

Configs compose from left to right; later files override single keys. Every model is trained in two stages:

1. 2D heatmap pre-training of the encoder at 256 × 256 (`configs/stage_heatmap.yaml`);
2. end-to-end training, initialised from the last stage-1 checkpoint.

`scripts/run_baselines.sh` runs both stages. It names the runs, writes logs and checkpoints to `work_dirs/<run>/`, and
records completion and the best-validation checkpoint (`scripts/ckpt_select.py`). The commands below reproduce the
configurations of the paper. Each configuration was trained with seeds 42, 1 and 2: append `--seed_everything <seed>`
to `EXTRA` and change `RUN_SUFFIX`. All runs used one GPU, except the ResNet-18 heatmap stage on UnrealEgo-RW
(4 GPUs × 64 clips; the commands below use 1 GPU × 256).

```bash
export ROOT=$DATA_ROOT/processed                    # checked by run_baselines.sh
D="--data.root $ROOT"
H="--data.batch_size 256 --trainer.strategy auto $D"                                   # stage 1
T="--trainer.strategy auto --data.batch_size 16 --data.clips_per_epoch_scale 2 $D"     # stage 2
S="--config configs/ablations/no_p4.yaml --config configs/enc_sfvit_s.yaml --config configs/invalid_view_mask.yaml"

# Stage 1, DINOv3 ViT-S+/16 encoder
HM_NAME=hm_sfvit_s_rw  TRAINSET=rw  GPUS=0 HM_EXTRA="--config configs/enc_sfvit_s.yaml $H" bash scripts/run_baselines.sh heatmap
HM_NAME=hm_sfvit_s_ue2 TRAINSET=ue2 GPUS=0 HM_EXTRA="--config configs/enc_sfvit_s.yaml $H" bash scripts/run_baselines.sh heatmap

# Table 1: trained on UnrealEgo-RW -> run egorayformer_device_rw_main
HM_NAME=hm_sfvit_s_rw TRAINSET=rw PROTOS=device GPUS=0 RUN_SUFFIX=_main \
  EXTRA="$S --config configs/res_384.yaml $T" bash scripts/run_baselines.sh egorayformer
# Table 1: trained on UnrealEgo2 (3 epochs) -> run egorayformer_device_ue2_main
HM_NAME=hm_sfvit_s_ue2 TRAINSET=ue2 PROTOS=device GPUS=0 RUN_SUFFIX=_main \
  EXTRA="$S --config configs/fov_circle.yaml --config configs/res_384.yaml --config configs/sched_e3.yaml $T" \
  bash scripts/run_baselines.sh egorayformer
# Table 1: UnrealEgo2, then UnrealEgo-RW (weights of the best UnrealEgo2 checkpoint of the same seed)
INIT_FROM_RUN=egorayformer_device_ue2_main TRAINSET=rw PROTOS=device GPUS=0 RUN_SUFFIX=_from_ue2 \
  EXTRA="$S --config configs/fov_circle.yaml --config configs/res_384.yaml --config configs/data_ue2_then_rw.yaml $T" \
  bash scripts/run_baselines.sh egorayformer
```

For the RW model, stage 2 expands to the following command (`DRY=1` prints the commands of a launcher without running
them):

```bash
python run.py fit --config configs/base_unrealego.yaml --config configs/data_device_relative.yaml \
  --config configs/res_256.yaml --config configs/data_rw_only.yaml --config configs/egorayformer_full.yaml \
  --model.encoder_pretrained "$(python scripts/ckpt_select.py last hm_sfvit_s_rw)" \
  --config configs/ablations/no_p4.yaml --config configs/enc_sfvit_s.yaml --config configs/invalid_view_mask.yaml \
  --config configs/res_384.yaml --trainer.strategy auto --data.batch_size 16 --data.clips_per_epoch_scale 2 \
  --data.root $ROOT --trainer.devices 1 --trainer.default_root_dir work_dirs/egorayformer_device_rw_main \
  --trainer.logger '[{"class_path":"pytorch_lightning.loggers.CSVLogger","init_args":{"save_dir":"work_dirs/egorayformer_device_rw_main"}}]'
```

The other experiments of the paper:

```bash
# Table 2: ResNet-18 + FPN at 256^2 (stage 1 with the default encoder)
HM_NAME=hm_r18_rw  TRAINSET=rw  GPUS=0 HM_EXTRA="$H" bash scripts/run_baselines.sh heatmap
HM_NAME=hm_r18_ue2 TRAINSET=ue2 GPUS=0 HM_EXTRA="$H" bash scripts/run_baselines.sh heatmap
R="--config configs/ablations/no_p4.yaml --config configs/invalid_view_mask.yaml"
HM_NAME=hm_r18_rw  TRAINSET=rw  PROTOS=device GPUS=0 RUN_SUFFIX=_r18 EXTRA="$R $T" bash scripts/run_baselines.sh egorayformer
HM_NAME=hm_r18_ue2 TRAINSET=ue2 PROTOS=device GPUS=0 RUN_SUFFIX=_r18 \
  EXTRA="$R --config configs/fov_circle.yaml --config configs/sched_e3.yaml $T" bash scripts/run_baselines.sh egorayformer

# Table 2: TSR-like, our implementation following the description of TSR-Ego (not a reproduction)
L="--config configs/tsr_layer_order.yaml --trainer.strategy auto --data.clips_per_epoch_scale 2 $D"
HM_NAME=hm_r18_rw  TRAINSET=rw  PROTOS=device GPUS=0 RUN_SUFFIX=_tsr EXTRA="$L" bash scripts/run_baselines.sh tsr
HM_NAME=hm_r18_ue2 TRAINSET=ue2 PROTOS=device GPUS=0 RUN_SUFFIX=_tsr EXTRA="$L" bash scripts/run_baselines.sh tsr
# Table 4 (protocol gap): the same TSR-like model with targets relative to the ground-truth mid-hip, which is also the
# projection origin in training and at inference (oracle root)
HM_NAME=hm_r18_rw  TRAINSET=rw  PROTOS=root   GPUS=0 RUN_SUFFIX=_tsr EXTRA="$L" bash scripts/run_baselines.sh tsr

# Table 3: ablations on UnrealEgo-RW (validation)
for abl in no_geo no_p3 no_track_prior; do        # geometric branch, trajectory-aligned sampling, track prior
  HM_NAME=hm_sfvit_s_rw TRAINSET=rw PROTOS=device GPUS=0 RUN_SUFFIX=_$abl \
    EXTRA="$S --config configs/res_384.yaml --config configs/ablations/$abl.yaml $T" bash scripts/run_baselines.sh egorayformer
done
# 256^2 input: the same without configs/res_384.yaml (full model and no_geo).
# Plain DINOv3 encoder: stage 1 and stage 2 with configs/enc_dinov3_plain_s.yaml in place of configs/enc_sfvit_s.yaml.
```

`EPF-device` in Table 2 was trained with the official [EgoPoseFormer](https://github.com/ChenhongyiYang/egoposeformer)
code and an UnrealEgo-RW adapter. The adapter is not part of this repository.

## Evaluation

Model selection uses the validation split only. `python scripts/ckpt_select.py best <run>` returns the
best-validation checkpoint of a finished run. `scripts/retest_runs.py` evaluates that checkpoint with the run's saved
configuration. The paper reports FP32 evaluation with one causal stream per sequence:

```bash
python scripts/retest_runs.py --from_config --runs egorayformer_device_rw_main --dataset unrealego_rw \
  --mode validate --gpus 0 --out work_dirs/eval --seq_dump \
  --extra "--model.eval_precision fp32 --data.eval_clip_len 0"
```

Metrics are written to `work_dirs/eval/<run>/lightning_logs/version_*/metrics.csv`. Use `--dataset unrealego2` for
UnrealEgo2 models. `--seq_dump` writes per-sequence error sums. `--diag_dump` writes the last-layer candidates
used for the error and gate analysis.

The test split is evaluated only for a frozen configuration. `run.py test` refuses to run unless
`EGORAYFORMER_ALLOW_TEST=1`, and `retest_runs.py --mode test` needs `--allow_test`.

## Repository layout

```
run.py                 LightningCLI entry point (fit / validate / test / predict)
egorayformer/
  data/                record format, clip dataset and protocols, fisheye camera models, heatmaps, skeletons
  models/              encoders (ResNet-18 + FPN, DINOv3 ConvNeXt, fisheye-native stereo ViT), attention and ray-geometry layers, EgoRayFormer
  training/            LightningModule, losses, typed config dataclasses
  eval/                metrics (MPJPE, PA-MPJPE, PCK, AUC, MPJVE, strata), input audit, benchmark export
configs/               composable configs: protocols, datasets, encoders, resolutions, model switches, ablations/
scripts/               data preparation, training launcher, checkpoint selection, evaluation
assets/                architecture figure
docs/logo/             logos
```

The code contains switches for further experiments that are off by default; each config file starts with a comment
that describes what it changes.

## Citation

```bibtex
@article{wang2026egorayformer,
  title   = {{EgoRayFormer}: Ray-Space Stereo Refinement for Egocentric {3D} Human Pose Estimation},
  author  = {Wang, Haozhe and Wu, Songyuan and Xu, Jihai and Zhao, Jie},
  journal = {arXiv preprint arXiv:2610.26641},
  year    = {2026}
}
```

## License and acknowledgements

This code is released under the [Apache License 2.0](LICENSE). Parts are adapted from other Apache-2.0 projects and
name their source at the top of the file:
* [EgoPoseFormer](https://github.com/ChenhongyiYang/egoposeformer): `models/backbones/r18_fpn.py`,
  `models/backbones/fpn.py`, `models/layers/deform_stereo_attn.py`, `models/layers/ray_stereo_attn.py` (module
  conventions), `data/camera.py`, `eval/metrics.py` (`batch_procrustes`);
* [mmcv](https://github.com/open-mmlab/mmcv): the sampling core of `models/layers/deform_stereo_attn.py`;
* [EgoRear](https://github.com/hiroyasuakada/EgoRear): `data/camera.py`.

The datasets and the DINOv3 weights are distributed under their own licenses and are not part of this repository.

---
## Contact Us

<p align="center">
  <img src="docs/logo/code.jpg" alt="WeChat QR code" width="110">
</p>

<p align="center">
  <a href="https://bodenai.com">bodenai.com</a> (中文)&nbsp;·&nbsp;
  <a href="https://www.boden.ai">boden.ai</a> (English)
</p>

<p align="center">
  <a href="mailto:company@bodenai.com">company@bodenai.com</a>&nbsp;·&nbsp;
  <a href="mailto:EAI@bodenai.com">EAI@bodenai.com</a>
</p>
