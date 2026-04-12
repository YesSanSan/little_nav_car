#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${PACKAGE_DIR}/models"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${MODELS_DIR}/.ultralytics-venv"
MODEL_NAME="${MODEL_NAME:-yolo26n.pt}"
MODEL_STEM="${MODEL_NAME%.pt}"
OUTPUT_DIR="${MODELS_DIR}/${MODEL_STEM}_ncnn_model"

mkdir -p "${MODELS_DIR}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install -U pip setuptools wheel
pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  ultralytics-opencv-headless \
  pnnx \
  ncnn

pushd "${MODELS_DIR}" >/dev/null
yolo export "model=${MODEL_NAME}" format=ncnn imgsz=640
popd >/dev/null

if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "Expected NCNN output directory not found: ${OUTPUT_DIR}" >&2
  exit 1
fi

echo "Exported ${MODEL_NAME} to ${OUTPUT_DIR}"
