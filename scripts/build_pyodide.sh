#!/usr/bin/env bash
# Build a Pyodide (WebAssembly) wheel of pyayay into dist-pyodide/.
#
# Defaults match Pyxel 2.9.4's web runtime: Pyodide 0.29.3, Python 3.13,
# emscripten 4.0.9. Override with PYODIDE_VERSION=... if needed.
# The cross-build env supplies -fwasm-exceptions, so pybind11's C++ exceptions
# work without extra flags.
set -euo pipefail

PYODIDE_VERSION="${PYODIDE_VERSION:-0.29.3}"
PYODIDE_BUILD_VERSION="${PYODIDE_BUILD_VERSION:-0.39.2}"
OUT_DIR="${OUT_DIR:-dist-pyodide}"

cd "$(dirname "$0")/.."

python -m pip install "pyodide-build==${PYODIDE_BUILD_VERSION}"
pyodide xbuildenv install "${PYODIDE_VERSION}"
pyodide xbuildenv install-emscripten

EMSDK_DIR="$(pyodide config get emsdk_dir)"
# shellcheck disable=SC1091
source "${EMSDK_DIR}/emsdk_env.sh" >/dev/null

# setuptools_scm cannot see a release tag on feature branches; pin the version
# from version.txt unless the caller pinned one already.
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-$(tr -d '[:space:]' < version.txt)}"

pyodide build --outdir "${OUT_DIR}"
git checkout -- version.txt 2>/dev/null || true
ls -l "${OUT_DIR}"
