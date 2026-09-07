#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMIR_DIR="${ROOT_DIR}/MimIR"
BUILD_DIR="${MIMIR_DIR}/build"
MLIR_PLUGIN_DIR="${MIMIR_DIR}/extra/mim_mlir"
MLIR_PLUGIN_URL="${MIM_MLIR_URL:-https://github.com/HarithF/mim_mlir.git}"

git -C "${ROOT_DIR}" submodule update --init --recursive

# MimIR globs extra/*/CMakeLists.txt at configure time, so the plugin has to be in place
# before the cmake call below for `mim_all_plugins` (and thus `mim_py`) to build it.
if [[ -d "${MLIR_PLUGIN_DIR}/.git" ]]; then
    echo "mim_mlir already checked out in ${MLIR_PLUGIN_DIR}; leaving it untouched"
else
    git clone "${MLIR_PLUGIN_URL}" "${MLIR_PLUGIN_DIR}"
fi

if [[ ! -d "${ROOT_DIR}/.venv" ]]; then
    uv venv --python 3.14
fi

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
elif [[ -x "${ROOT_DIR}/.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/Scripts/python.exe"
else
    echo "Could not find Python in ${ROOT_DIR}/.venv" >&2
    exit 1
fi

cmake -S "${MIMIR_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${MIMIR_BUILD_TYPE:-Release}" \
    -DMIM_BUILD_PYTHON=ON \
    -DPython_EXECUTABLE="${PYTHON_BIN}"
cmake --build "${BUILD_DIR}" --target mim_py -j "${MIMIR_BUILD_JOBS:-8}"

cd "${ROOT_DIR}"
uv sync
