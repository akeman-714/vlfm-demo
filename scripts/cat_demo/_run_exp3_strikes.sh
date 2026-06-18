#!/usr/bin/env bash
# exp3 重跑: coco=0.5 + update_explored 5帧宽限(VLFM_EXPLORED_STRIKES) + dbscan min_points 放宽
# 目标: 远处检到橘猫先保留+走过去, 抵近 5 帧内确认到就锁定, 不再"进锥秒删"。
set -u
cd /data/jinsong.yuan/vlfm-demo/vlfm
TS="$(date +%Y%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES=2,7
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost

export VIDEO_DIR=video_dir/exp3_strikes_orangecat_gpu2_${TS}
export LOG=outputs/exp3_strikes_orangecat_gpu2_${TS}.log
export TB_DIR=tb/exp3_strikes_orangecat_gpu2_${TS}

# --- exp3 原样 env ---
export VLFM_OBJECTNAV_QUERY=找橘猫
export VLFM_ATTR_NOUN=cat
export VLFM_ATTR_PREDICATE="an orange cat"
export VLFM_ATTR_USE_VALUE_TEXT=1
export VLFM_ATTR_VERIFY=1
export VLFM_ATTR_FAIL_OPEN=0
export VLFM_ATTR_VERIFY_TIMEOUT=30
export ATTR_VERIFIER_PORT=12187

# --- 本次变量 ---
export VLFM_COCO_THRESHOLD=0.5        # 已验证: 让橘猫检测能进
export VLFM_EXPLORED_STRIKES=5        # 抵近 5 帧宽限再删 (核心改动)
export VLFM_DBSCAN_MIN_POINTS=50      # 远处稀疏点云也允许入图

echo ">>> exp3 strikes | coco=${VLFM_COCO_THRESHOLD} strikes=${VLFM_EXPLORED_STRIKES} dbscan_min=${VLFM_DBSCAN_MIN_POINTS} | video=${VIDEO_DIR}"
exec bash scripts/cat_demo/eval_cat_demo.sh
