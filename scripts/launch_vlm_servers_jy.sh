#!/usr/bin/env bash
# 自定义版：在 conda env (vlfm_pip) 中启动 4 个 VLM Flask 服务，全部固定到 GPU 0。
# 用法：bash scripts/launch_vlm_servers_jy.sh [GPU_ID]
# 默认 GPU=0；改成别的卡号即可。
set -euo pipefail

GPU_ID="${1:-0}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-vlfm_pip}"

export MOBILE_SAM_CHECKPOINT="${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}"
export GROUNDING_DINO_CONFIG="${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
export GROUNDING_DINO_WEIGHTS="${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}"
export CLASSES_PATH="${CLASSES_PATH:-vlfm/vlm/classes.txt}"
export GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-12181}"
export BLIP2ITM_PORT="${BLIP2ITM_PORT:-12182}"
export SAM_PORT="${SAM_PORT:-12183}"
export YOLOV7_PORT="${YOLOV7_PORT:-12184}"

# 国内访问 huggingface.co 不通时走镜像
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

session_name="vlm_servers_${RANDOM}"

# 每个 pane 启动前要做的事：cd 仓库 -> source conda -> activate -> pin GPU
prefix="cd ${REPO_DIR} && source ${CONDA_SH} && conda activate ${CONDA_ENV} && export CUDA_VISIBLE_DEVICES=${GPU_ID} && export HF_ENDPOINT='${HF_ENDPOINT}' && export MOBILE_SAM_CHECKPOINT='${MOBILE_SAM_CHECKPOINT}' && export GROUNDING_DINO_CONFIG='${GROUNDING_DINO_CONFIG}' && export GROUNDING_DINO_WEIGHTS='${GROUNDING_DINO_WEIGHTS}' && export CLASSES_PATH='${CLASSES_PATH}'"

tmux new-session -d -s "${session_name}"
tmux split-window -v -t "${session_name}:0"
tmux split-window -h -t "${session_name}:0.0"
tmux split-window -h -t "${session_name}:0.2"

tmux send-keys -t "${session_name}:0.0" "${prefix} && python -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t "${session_name}:0.1" "${prefix} && python -m vlfm.vlm.blip2itm     --port ${BLIP2ITM_PORT}"      C-m
tmux send-keys -t "${session_name}:0.2" "${prefix} && python -m vlfm.vlm.sam           --port ${SAM_PORT}"           C-m
tmux send-keys -t "${session_name}:0.3" "${prefix} && python -m vlfm.vlm.yolov7        --port ${YOLOV7_PORT}"        C-m

echo "Started tmux session: ${session_name}  (GPU ${GPU_ID})"
echo "Ports: GDINO=${GROUNDING_DINO_PORT}  BLIP2ITM=${BLIP2ITM_PORT}  SAM=${SAM_PORT}  YOLOv7=${YOLOV7_PORT}"
echo "Attach:  tmux attach -t ${session_name}"
echo "Kill:    tmux kill-session -t ${session_name}"
echo "Wait ~60-90s for weights to load before running vlfm.run."
