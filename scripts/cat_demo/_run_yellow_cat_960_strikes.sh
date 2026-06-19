#!/usr/bin/env bash
# yellow-cat rerun with the same anti-flicker/object-map knobs that fixed the
# orange-cat run. Defaults remain unchanged elsewhere; this script is opt-in.
set -u

cd /data/jinsong.yuan/vlfm-demo/vlfm
TS="$(date +%Y%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,7}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"

export VIDEO_DIR="${VIDEO_DIR:-/data/jinsong.yuan/vlfm-demo/vlfm/video_dir/yellow_cat_960_strikes_${TS}}"
export LOG="${LOG:-/data/jinsong.yuan/vlfm-demo/vlfm/outputs/yellow_cat_960_strikes_${TS}.log}"
export TB_DIR="${TB_DIR:-/data/jinsong.yuan/vlfm-demo/vlfm/tb/yellow_cat_960_strikes_${TS}}"

export VLFM_OBJECTNAV_QUERY="${VLFM_OBJECTNAV_QUERY:-找黄色猫}"
export VLFM_ATTR_NOUN="${VLFM_ATTR_NOUN:-cat}"
export VLFM_ATTR_PREDICATE="${VLFM_ATTR_PREDICATE:-a yellow cat}"
export VLFM_ATTR_USE_VALUE_TEXT="${VLFM_ATTR_USE_VALUE_TEXT:-1}"
export VLFM_ATTR_VERIFY="${VLFM_ATTR_VERIFY:-1}"
export VLFM_ATTR_FAIL_OPEN="${VLFM_ATTR_FAIL_OPEN:-0}"
export VLFM_ATTR_VERIFY_TIMEOUT="${VLFM_ATTR_VERIFY_TIMEOUT:-30}"
export VLFM_ATTR_MAX_VERIFY_CALLS="${VLFM_ATTR_MAX_VERIFY_CALLS:-5}"
export ATTR_VERIFIER_PORT="${ATTR_VERIFIER_PORT:-12187}"

export VLFM_COCO_THRESHOLD="${VLFM_COCO_THRESHOLD:-0.5}"
export VLFM_EXPLORED_STRIKES="${VLFM_EXPLORED_STRIKES:-5}"
export VLFM_DBSCAN_MIN_POINTS="${VLFM_DBSCAN_MIN_POINTS:-50}"

echo ">>> yellow-cat 960 strikes | coco=${VLFM_COCO_THRESHOLD} strikes=${VLFM_EXPLORED_STRIKES} dbscan_min=${VLFM_DBSCAN_MIN_POINTS} value_text=${VLFM_ATTR_USE_VALUE_TEXT} | video=${VIDEO_DIR}"
exec bash scripts/cat_demo/eval_yellow_cat_demo.sh
