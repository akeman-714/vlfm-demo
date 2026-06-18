#!/usr/bin/env bash
# Run the attribute-verification demo:
#   user request "找黄色猫" -> noun "cat" drives detection/navigation ->
#   arrival crop is verified against "yellow cat" before STOP is allowed.
#
# Prerequisites:
#   ENABLE_ATTR_VERIFIER=1 SIGLIP_FORM=full bash scripts/launch_vlm_servers_jy.sh <vlm_gpu>
#   export BAILIAN_API_KEY=...   # read by the verifier server; never stored here
#
# Keep Habitat's renderer on an empty card.  The underlying eval_cat_demo.sh treats
# CUDA_VISIBLE_DEVICES=a,b as sim=cuda:0 (physical a), torch=cuda:1 (physical b).
set -u

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
export VLFM_OBJECTNAV_QUERY="${VLFM_OBJECTNAV_QUERY:-找黄色猫}"
export VLFM_ATTR_VERIFY="${VLFM_ATTR_VERIFY:-1}"
export VLFM_ATTR_USE_VALUE_TEXT="${VLFM_ATTR_USE_VALUE_TEXT:-0}"
export VLFM_ATTR_VERIFY_TIMEOUT="${VLFM_ATTR_VERIFY_TIMEOUT:-20.0}"
export VLFM_ATTR_MAX_VERIFY_CALLS="${VLFM_ATTR_MAX_VERIFY_CALLS:-4}"
export SPLIT="${SPLIT:-cat_demo}"
export VIDEO_DIR="${VIDEO_DIR:-video_dir/yellow_cat_${TS}}"
export TB_DIR="${TB_DIR:-tb/yellow_cat_${TS}}"
export LOG="${LOG:-outputs/yellow_cat_${TS}.log}"

echo ">>> Attribute cat demo"
echo ">>> request=${VLFM_OBJECTNAV_QUERY}"
echo ">>> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,7}  (sim first, torch second)"
echo ">>> video_dir=${VIDEO_DIR}"
echo ">>> log=${LOG}"
echo

echo "-- selected GPU usage before sim launch --"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | head -1
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tail -n +2 | awk -F',' -v visible="${CUDA_VISIBLE_DEVICES:-0,7}" '
  BEGIN {
    split(visible, gpu_ids, ",")
    for (i in gpu_ids) {
      gsub(/ /, "", gpu_ids[i])
      want[gpu_ids[i]] = 1
    }
  }
  { gsub(/ /,""); idx=$1; if (idx in want) print "  GPU "$1": used="$2" free="$3 }'
echo

exec bash scripts/cat_demo/eval_cat_demo.sh
