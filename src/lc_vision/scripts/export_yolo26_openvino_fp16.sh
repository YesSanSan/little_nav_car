#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${PACKAGE_DIR}/models"
WEIGHTS_DIR="${MODELS_DIR}/weights"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${MODELS_DIR}/.ultralytics-venv"
MODEL_NAME="${MODEL_NAME:-yolo26n.pt}"
MODEL_STEM="${MODEL_NAME%.pt}"
MODEL_PATH="${WEIGHTS_DIR}/${MODEL_NAME}"
TEMP_OUTPUT_DIR="${WEIGHTS_DIR}/${MODEL_STEM}_openvino_model"
OUTPUT_DIR="${MODELS_DIR}/${MODEL_STEM}_openvino_model"

mkdir -p "${MODELS_DIR}" "${WEIGHTS_DIR}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model weights not found: ${MODEL_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install -U pip setuptools wheel
pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  ultralytics \
  openvino

rm -rf "${TEMP_OUTPUT_DIR}" "${OUTPUT_DIR}"

pushd "${MODELS_DIR}" >/dev/null
yolo export "model=weights/${MODEL_NAME}" format=openvino half=True imgsz=640
popd >/dev/null

if [[ ! -d "${TEMP_OUTPUT_DIR}" ]]; then
  echo "Expected OpenVINO output directory not found: ${TEMP_OUTPUT_DIR}" >&2
  exit 1
fi

mv "${TEMP_OUTPUT_DIR}" "${OUTPUT_DIR}"

if [[ ! -f "${OUTPUT_DIR}/metadata.yaml" ]]; then
  echo "Expected metadata file not found: ${OUTPUT_DIR}/metadata.yaml" >&2
  exit 1
fi

echo "Exported ${MODEL_NAME} to ${OUTPUT_DIR} as OpenVINO FP16"
