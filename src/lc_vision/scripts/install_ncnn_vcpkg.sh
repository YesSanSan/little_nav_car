#!/usr/bin/env bash
set -euo pipefail

VCPKG_DIR="${HOME}/vcpkg"

if [[ ! -d "${VCPKG_DIR}" ]]; then
  git clone --depth 1 https://github.com/microsoft/vcpkg.git "${VCPKG_DIR}"
fi

if [[ ! -x "${VCPKG_DIR}/vcpkg" ]]; then
  "${VCPKG_DIR}/bootstrap-vcpkg.sh"
fi

"${VCPKG_DIR}/vcpkg" install ncnn:arm64-linux

echo "ncnn installed via vcpkg at ${VCPKG_DIR}"
