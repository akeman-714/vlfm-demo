#!/usr/bin/env bash
# 自定义版：启动 4 个 VLM Flask 服务，全部固定到 GPU 0。
#   - GDINO / BLIP2ITM / SAM 跑在 vlfm_pip 环境。
#   - YOLO(COCO 检测) 已换成 YOLO26+TensorRT，跑在独立的 yolo_trt 环境(端口不变 12184)。
# 用法：bash scripts/launch_vlm_servers_jy.sh [GPU_ID]
# 默认 GPU=0；改成别的卡号即可。
# 回滚到老 YOLOv7：把 0.3 pane 的 yolo_prefix/yolo_trt 换回 ${prefix} + vlfm.vlm.yolov7。
#
# ITM 后端切换(默认 siglip2,协议完全兼容,下游 policy/value_map/reality 都不用改)：
#   bash scripts/launch_vlm_servers_jy.sh 0                     # 默认:SigLIP2-base ITM(siglip2_itm)
#   ITM_BACKEND=blip2 bash scripts/launch_vlm_servers_jy.sh 0   # 回滚:BLIP2-ITM(vlfm_pip)
set -euo pipefail

GPU_ID="${1:-0}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-vlfm_pip}"
# YOLO26+TensorRT 检测器跑在自己的 env(ultralytics+tensorrt),不污染 vlfm_pip。
YOLO_TRT_ENV="${YOLO_TRT_ENV:-yolo_trt}"
YOLO_TRT_MODEL="${YOLO_TRT_MODEL:-data/yolo26l.engine}"

# ITM(图文相似度,value map 用)后端可切换：默认 siglip2;blip2 仍可一键回滚。
ITM_BACKEND="${ITM_BACKEND:-siglip2}"
SIGLIP_CONDA_ENV="${SIGLIP_CONDA_ENV:-siglip2_itm}"
DEFAULT_SIGLIP_MODEL_ID="/data/jinsong.yuan/siglip2-base-patch16-384"
if [ ! -d "${DEFAULT_SIGLIP_MODEL_ID}" ]; then
  DEFAULT_SIGLIP_MODEL_ID="google/siglip2-base-patch16-384"
fi
SIGLIP_MODEL_ID="${SIGLIP_MODEL_ID:-${DEFAULT_SIGLIP_MODEL_ID}}"

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

# YOLO pane 用 yolo_trt env；PYTHONPATH 让 `python -m vlfm.vlm.yolo_trt` 不必 pip 装 vlfm 也能找到包。
yolo_prefix="cd ${REPO_DIR} && source ${CONDA_SH} && conda activate ${YOLO_TRT_ENV} && export CUDA_VISIBLE_DEVICES=${GPU_ID} && export PYTHONPATH=${REPO_DIR}"

# SigLIP2 pane 用独立 env + PYTHONPATH(不必把整个 vlfm pip 装进新 env),避免污染 vlfm_pip 的 transformers==4.26.0。
siglip_prefix="cd ${REPO_DIR} && source ${CONDA_SH} && conda activate ${SIGLIP_CONDA_ENV} && export CUDA_VISIBLE_DEVICES=${GPU_ID} && export PYTHONPATH=${REPO_DIR} && export HF_ENDPOINT='${HF_ENDPOINT}' && export SIGLIP_MODEL_ID='${SIGLIP_MODEL_ID}'"

# 按 ITM_BACKEND 选 12182 pane 的命令。校验放在建 tmux session 之前,避免留下半启动的会话。
case "${ITM_BACKEND}" in
  blip2)
    itm_cmd="${prefix} && python -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}"
    ;;
  siglip2)
    itm_cmd="${siglip_prefix} && python -m vlfm.vlm.siglip2itm --port ${BLIP2ITM_PORT}"
    ;;
  *)
    echo "Unknown ITM_BACKEND=${ITM_BACKEND}; expected blip2 or siglip2" >&2
    exit 1
    ;;
esac

tmux new-session -d -s "${session_name}"
tmux split-window -v -t "${session_name}:0"
tmux split-window -h -t "${session_name}:0.0"
tmux split-window -h -t "${session_name}:0.2"

tmux send-keys -t "${session_name}:0.0" "${prefix} && python -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t "${session_name}:0.1" "${itm_cmd}" C-m
tmux send-keys -t "${session_name}:0.2" "${prefix} && python -m vlfm.vlm.sam           --port ${SAM_PORT}"           C-m
tmux send-keys -t "${session_name}:0.3" "${yolo_prefix} && python -m vlfm.vlm.yolo_trt --port ${YOLOV7_PORT} --model ${YOLO_TRT_MODEL}" C-m

echo "Started tmux session: ${session_name}  (GPU ${GPU_ID})"
echo "ITM backend: ${ITM_BACKEND}  (port ${BLIP2ITM_PORT}, route /blip2itm)"
if [ "${ITM_BACKEND}" = "siglip2" ]; then
  echo "  SigLIP2 env=${SIGLIP_CONDA_ENV}  model=${SIGLIP_MODEL_ID}"
fi
echo "Ports: GDINO=${GROUNDING_DINO_PORT}  BLIP2ITM=${BLIP2ITM_PORT}  SAM=${SAM_PORT}  YOLO26-TRT=${YOLOV7_PORT}"
echo "Attach:  tmux attach -t ${session_name}"
echo "Kill:    tmux kill-session -t ${session_name}"
echo "Wait ~60-90s for weights to load before running vlfm.run."
