#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
MASTER_PORT="${MASTER_PORT:-29512}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/train/rainsyn_light25}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${PROJECT_DIR}/options/rainsyn_light25.json}"
GENERATED_CONFIG="${GENERATED_CONFIG:-${OUTPUT_DIR}/config.json}"
TRAIN_TXT="${TRAIN_TXT:-${OUTPUT_DIR}/lists/train.txt}"
TEST_TXT="${TEST_TXT:-${OUTPUT_DIR}/lists/test.txt}"

: "${TRAIN_H5:?Set TRAIN_H5 to the RainSynLight25 training H5 path.}"
: "${TEST_H5:?Set TEST_H5 to the RainSynLight25 test H5 path.}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/prepare_rainsyn_lists.py" \
  --train-h5 "${TRAIN_H5}" \
  --test-h5 "${TEST_H5}" \
  --train-out "${TRAIN_TXT}" \
  --test-out "${TEST_TXT}" \
  --sequence-length 3

"${PYTHON_BIN}" "${PROJECT_DIR}/prepare_release_config.py" \
  --base-config "${CONFIG_TEMPLATE}" \
  --output-config "${GENERATED_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --train-h5 "${TRAIN_H5}" \
  --test-h5 "${TEST_H5}" \
  --train-txt "${TRAIN_TXT}" \
  --test-txt "${TEST_TXT}"

(
  cd "${PROJECT_DIR}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT="${MASTER_PORT}"
  export WORLD_SIZE=1
  export RANK=0
  "${PYTHON_BIN}" main.py --config "${GENERATED_CONFIG}" --local-rank 0
)
