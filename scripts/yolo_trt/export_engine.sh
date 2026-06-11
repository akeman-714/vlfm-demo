#!/usr/bin/env bash
# Phase 1 of docs/notes_zh/09_YOLO26_TensorRT替换方案.md
#
# Download <model>.pt and compile a TensorRT FP16 engine ON THIS BOX.
# Default model is yolo26n; set YOLO_TRT_WEIGHTS=yolo26l.pt (or yolo26m.pt) to
# switch variants -- the engine filename is derived from it (see ENGINE below).
# !! The engine binds to the GPU arch (H20 = sm_90) + TRT version + CUDA, so it
#    is NOT portable. Always (re)export on the machine that will run it (pit #1).
#
# Output: data/<model>.pt (weights) and data/<model>.engine (TRT engine), kept
# alongside the other VLM checkpoints. The launch script loads the .engine.
set -euo pipefail

ENV_NAME="${YOLO_TRT_ENV:-yolo_trt}"
CONDA_SH="${CONDA_SH:-/data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${YOLO_TRT_WEIGHTS:-yolo26n.pt}"
ENGINE="${MODEL%.pt}.engine"  # yolo26n.pt -> yolo26n.engine; tracks YOLO_TRT_WEIGHTS
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"

export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"
cd "${REPO_DIR}/data"  # weights + engine live next to the other checkpoints

# Export FP16 engine (auto-downloads ${MODEL} via proxy on first run).
yolo export model="${MODEL}" format=engine half=True imgsz="${IMGSZ}" device="${DEVICE}"

echo "=== engine written (data/${ENGINE}) ==="
ls -lh "${REPO_DIR}/data/${ENGINE}"

# --- Optional acceptance gates (G1.2 accuracy, G1.3 speed) ---------------------
# Downloads COCO val2017 (~1GB) on first run; uncomment to run.
# yolo val       model="${ENGINE}" data=coco.yaml imgsz="${IMGSZ}" device="${DEVICE}"  # FP16 engine mAP
# yolo val       model="${MODEL}"  data=coco.yaml imgsz="${IMGSZ}" device="${DEVICE}"  # .pt baseline mAP
# yolo benchmark model="${ENGINE}" imgsz="${IMGSZ}" device="${DEVICE}"                 # ms/frame
