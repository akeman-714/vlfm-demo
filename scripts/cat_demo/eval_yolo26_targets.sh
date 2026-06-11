#!/usr/bin/env bash
# Run the yolo26+TensorRT detector against the cat_demo scene (TEEsavR23oF) for
# two COCO targets in sequence: refrigerator, then toilet. Reuses eval_cat_demo.sh
# but overrides the goal via VLFM_GOAL_SEQUENCE (single-target ordered plan).
#
# Topology (1b, validated): sim renderer on a CUDA-free card, nav+VLM on GPU0.
#   CUDA_VISIBLE_DEVICES=<free>,0  -> cuda:0 = sim (free card), cuda:1 = torch (GPU0).
# Proxy: localhost VLM calls die behind the shell's HTTP_PROXY unless no_proxy is set.
set -u

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,0}"
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

TS="$(date +%Y%m%d_%H%M%S)"
TIMEOUT_S="${TIMEOUT_S:-1800}"
# Space-separated COCO goals, run in sequence. Override e.g. TARGETS="cat toilet chair".
TARGETS="${TARGETS:-refrigerator toilet}"

run_target() {  # $1 = goal label (COCO name)
  local goal="$1"
  local vdir="${REPO_DIR}/video_dir/yolo26_${goal}_${TS}"
  echo "=============================================================="
  echo ">>> yolo26 target='${goal}'  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo ">>> video_dir=${vdir}"
  echo "=============================================================="
  VLFM_GOAL_SEQUENCE="${goal}" \
  VIDEO_DIR="${vdir}" \
  TB_DIR="${REPO_DIR}/tb/yolo26_${goal}_${TS}" \
  LOG="${REPO_DIR}/outputs/yolo26_${goal}_${TS}.log" \
  timeout "${TIMEOUT_S}" bash scripts/cat_demo/eval_cat_demo.sh
  echo "[${goal}] eval_cat_demo exit=$?"
  echo "[${goal}] videos:"; ls -1 "${vdir}"/*.mp4 2>/dev/null || echo "  (none)"
}

for goal in ${TARGETS}; do
  run_target "${goal}"
done

echo "=============================================================="
echo "ALL_DONE  -- absolute video paths:"
for goal in ${TARGETS}; do
  for f in "${REPO_DIR}/video_dir/yolo26_${goal}_${TS}"/*.mp4; do
    [ -e "$f" ] && echo "  ${goal}: $f"
  done
done
