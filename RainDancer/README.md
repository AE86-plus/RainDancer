# RainDancer

PyTorch implementation of the released RainDancer code path used for NTURain-v2e, RainSynComplex25, and RainSynLight25.

## Scope

This open-source release keeps only the code for:

- NTURain-v2e
- RainSynComplex25
- RainSynLight25

The released test setting uses:

- crop size: `128`
- stride: `64`

## Repository Layout

- `main.py`: training entry point
- `train_nturain_v2e.sh`: NTURain-v2e training helper
- `train_rainsyn_complex25.sh`: RainSynComplex25 training helper
- `train_rainsyn_light25.sh`: RainSynLight25 training helper
- `test_nturain_v2e.py`: NTURain-v2e test script
- `test_rainsyn_complex25.py`: RainSynComplex25 test script
- `test_rainsyn_light25.py`: RainSynLight25 test script
- `infer_bgr_rgb_sliding_common.py`: shared sliding-window inference utilities
- `prepare_release_config.py`: helper to generate runnable configs with user paths
- `options/`: config templates
- `weights/`: optional local directory for your checkpoints

## Environment

Recommended environment:

- Python 3.8
- PyTorch with CUDA support

Install dependencies with:

```bash
pip install -r requirements.txt
```

`mmcv` must provide `mmcv.ops.DeformConv2d` for your CUDA and PyTorch version.

## Data and Checkpoints

This release does not assume your dataset paths.

You should provide:

- your own training and test H5 paths
- your own checkpoint path when testing

This GitHub repository does not include pretrained weights.

If you want to follow the suggested local layout, place your checkpoints at:

- `weights/nturain/best.pth.tar`
- `weights/rainsyn-complex25/best.pth.tar`
- `weights/rainsyn-light25/best.pth.tar`

You can also keep checkpoints anywhere else and pass the path with `--checkpoint`.

## Training Commands

All training scripts write outputs under `outputs/train/...` by default.

### NTURain-v2e

```bash
TRAIN_H5=/path/to/NTURain-v2e/train.h5 \
TEST_H5=/path/to/NTURain-v2e/test.h5 \
GPU_ID=0 PYTHON_BIN=python \
bash ./train_nturain_v2e.sh
```

Optional variables:

- `OUTPUT_DIR=/path/to/output_dir`
- `MASTER_PORT=29501`
- `TRAIN_TXT=/path/to/train.txt`
- `TEST_TXT=/path/to/test.txt`

### RainSynComplex25

```bash
TRAIN_H5=/path/to/RainSynComplex25-train.h5 \
TEST_H5=/path/to/RainSynComplex25-test.h5 \
GPU_ID=0 PYTHON_BIN=python \
bash ./train_rainsyn_complex25.sh
```

Optional variables:

- `OUTPUT_DIR=/path/to/output_dir`
- `MASTER_PORT=29511`

### RainSynLight25

```bash
TRAIN_H5=/path/to/RainSynLight25-train.h5 \
TEST_H5=/path/to/RainSynLight25-test.h5 \
GPU_ID=0 PYTHON_BIN=python \
bash ./train_rainsyn_light25.sh
```

Optional variables:

- `OUTPUT_DIR=/path/to/output_dir`
- `MASTER_PORT=29512`

## Test Commands

All test scripts write outputs under `outputs/inference/...` by default.

### NTURain-v2e

```bash
python test_nturain_v2e.py \
  --gpu 0 \
  --test-h5 /path/to/NTURain-v2e/test.h5 \
  --checkpoint /path/to/checkpoint.pth.tar
```

### RainSynComplex25

```bash
python test_rainsyn_complex25.py \
  --gpu 0 \
  --test-h5 /path/to/RainSynComplex25-test.h5 \
  --checkpoint /path/to/checkpoint.pth.tar
```

### RainSynLight25

```bash
python test_rainsyn_light25.py \
  --gpu 0 \
  --test-h5 /path/to/RainSynLight25-test.h5 \
  --checkpoint /path/to/checkpoint.pth.tar
```

Useful optional arguments for all three test scripts:

- `--output-root /path/to/output_dir`
- `--crop-size 128`
- `--stride 64`
- `--patch-batch 4`
- `--save-images 1`
- `--max-scenes 0`
- `--max-samples-per-scene 0`

## Output Directories

Default output locations:

- training: `outputs/train/<dataset_name>/`
- testing: `outputs/inference/<dataset_name>/`

Each test run writes:

- `summary.json`
- `summary.txt`
- `sample_metrics.csv`
- `scene_metrics.csv`
- optional saved images if `--save-images 1`

## Notes

- `NTURain-v2e_real` is not included in this release workflow.
- The old combined multi-dataset test entry was removed in favor of per-dataset scripts.
- Historical auxiliary scripts, cached outputs, and unrelated experimental code were removed.
