#!/usr/bin/env bash
# Phase 0 of docs/notes_zh/09_YOLO26_TensorRT替换方案.md
#
# Build the standalone `yolo_trt` conda env for the YOLO26 + TensorRT COCO
# detector. Kept SEPARATE from vlfm_cuda_sim / vlfm_pip so tensorrt/ultralytics
# cannot clobber the simulator's pinned torch (pit #0). The detector talks to
# VLFM over HTTP, so it need not share an env with the policy.
#
# Lessons baked in (this box: 8x H20-3e sm_90, driver 570.172.08 = CUDA 12.8):
#   * The local proxy (7897) drops multi-GB wheels mid-stream -> pull from
#     domestic mirrors WITHOUT the proxy instead. (The proxy is only needed
#     later, for the GitHub weight download in export_engine.sh.)
#   * Default PyPI now ships a cu130 torch + cu13 TensorRT; on a CUDA-12.8 driver
#     those load but report `torch.cuda.is_available() == False`. We must pin the
#     cu128 torch wheels and the cu12 TensorRT build.
#   * This box's ~/.config/pip/pip.conf adds a PyPI mirror as extra-index (which
#     serves the cu130 torch and, being a higher version, wins). PIP_CONFIG_FILE
#     =/dev/null neutralizes it so the single --index-url we pass is authoritative.
#
# Re-runnable: safe to `conda env remove -n yolo_trt` and run again.
set -uo pipefail  # NOT -e: each layer's import is verified explicitly below

ENV_NAME="${YOLO_TRT_ENV:-yolo_trt}"
CONDA_SH="${CONDA_SH:-/data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Domestic mirrors, proxy OFF, global pip.conf ignored.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export no_proxy='*' NO_PROXY='*'
export PIP_CONFIG_FILE=/dev/null
PYPI="${PYPI:-https://pypi.tuna.tsinghua.edu.cn/simple}"            # general packages
TORCH_IDX="${TORCH_IDX:-https://mirror.nju.edu.cn/pytorch/whl/cu128}"  # cu128 torch (driver 12.8)
PIPX="pip install --retries 15 --timeout 120"

source "${CONDA_SH}"
conda create -n "${ENV_NAME}" python=3.11 -y
conda activate "${ENV_NAME}"
$PIPX --index-url "${PYPI}" --upgrade pip

# 1) cu128 torch/torchvision -- MUST match the CUDA 12.8 driver; cu130 won't see the GPU.
$PIPX --index-url "${TORCH_IDX}" torch torchvision
# 2) ultralytics + onnx export path + flask server deps. onnxruntime-gpu is pulled
#    here so `yolo export ... format=engine` does not auto-install it through the
#    proxy at build time (that uv AutoUpdate stalls on the proxy).
$PIPX --index-url "${PYPI}" ultralytics onnx onnxslim onnxruntime-gpu flask requests
# 3) TensorRT cu12 build (cu13 needs a newer driver). Pin to an ultralytics-supported 10.x.
$PIPX --index-url "${PYPI}" "tensorrt-cu12==10.9.0.34"

# Verify GPU visibility + all imports before declaring success.
python - <<'PY'
import torch, tensorrt, ultralytics, cv2
assert torch.cuda.is_available(), "torch cannot see the GPU -- wrong CUDA build for this driver"
print("torch       ", torch.__version__, "(cuda", torch.version.cuda, "dev", torch.cuda.get_device_name(0) + ")")
print("tensorrt    ", tensorrt.__version__)
print("ultralytics ", ultralytics.__version__, "| cv2", cv2.__version__)
PY

# Pin everything: the engine export env MUST equal the runtime env (pit #1).
pip freeze > "${SCRIPT_DIR}/requirements-pinned.txt"
echo "=== yolo_trt env ready (pinned -> requirements-pinned.txt) ==="
