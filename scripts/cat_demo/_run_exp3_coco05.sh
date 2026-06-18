#!/usr/bin/env bash
# exp3 重跑 (SigLIP2 找 orange cat, 目标橘猫) —— 唯一改动: coco_threshold 0.8 -> 0.5
# 复刻 exp3_retry2 的 env, 只降 YOLO 判定阈值, 验证"橘猫检测过不了 0.8"假设。
set -u
cd /data/jinsong.yuan/vlfm-demo/vlfm
TS="$(date +%Y%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES=2,7              # sim=GPU2, torch=GPU7 (与 exp3 一致)
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost

export VIDEO_DIR=video_dir/exp3_coco05_orangetext_orangecat_gpu2_${TS}
export LOG=outputs/exp3_coco05_orangetext_orangecat_gpu2_${TS}.log
export TB_DIR=tb/exp3_coco05_orangetext_orangecat_gpu2_${TS}

# --- exp3 原样 env ---
export VLFM_OBJECTNAV_QUERY=找橘猫
export VLFM_ATTR_NOUN=cat
export VLFM_ATTR_PREDICATE="an orange cat"
export VLFM_ATTR_USE_VALUE_TEXT=1           # SigLIP value text = "orange cat"
export VLFM_ATTR_VERIFY=1
export VLFM_ATTR_FAIL_OPEN=0
export VLFM_ATTR_VERIFY_TIMEOUT=30          # 长超时, 避免掉回 heuristic
export ATTR_VERIFIER_PORT=12187             # 长超时 Qwen verifier

# --- 本次唯一变量 ---
export VLFM_COCO_THRESHOLD=0.5

echo ">>> exp3 coco05 | coco_threshold=${VLFM_COCO_THRESHOLD} | predicate='${VLFM_ATTR_PREDICATE}' | verifier=${ATTR_VERIFIER_PORT} | video=${VIDEO_DIR}"
exec bash scripts/cat_demo/eval_cat_demo.sh
