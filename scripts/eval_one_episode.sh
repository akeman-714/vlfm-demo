#!/usr/bin/env bash
# 跑 1 条 ObjectNav HM3D val 集，并把第三人称视频存到 disk。
# 前置：scripts/launch_vlm_servers_jy.sh 已经把 4 个 VLM Flask 服务跑起来。
# 用法：bash scripts/eval_one_episode.sh           # 跑 1 集
#       bash scripts/eval_one_episode.sh 5         # 跑 5 集
#       N_EP=3 GPU_ID=1 bash scripts/eval_one_episode.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_pip

N_EP="${1:-${N_EP:-1}}"
GPU_ID="${GPU_ID:-7}"          # eval 进程跑哪张卡。GPU 7 最空（注意：BLIP2-ITM 也在 7，~3GB；其他 3 个 VLM 在 GPU 0）
SPLIT="${SPLIT:-val}"          # 也可以试 val_mini（数据少）
VIDEO_DIR="${VIDEO_DIR:-video_dir/itm_$(date +%Y%m%d_%H%M%S)}"
TB_DIR="${TB_DIR:-tb/itm_$(date +%Y%m%d_%H%M%S)}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${VIDEO_DIR}" "${TB_DIR}"

echo ">>> eval split=${SPLIT} N_EP=${N_EP} GPU=${GPU_ID}"
echo ">>> video_dir=${VIDEO_DIR}"

python -um vlfm.run \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=data/dummy_policy.pth \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.rl.policy.name=HabitatITMPolicyV2 \
  habitat.task.lab_sensors.base_explorer.turn_angle=30 \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.split="${SPLIT}" \
  habitat_baselines.video_dir="${VIDEO_DIR}" \
  habitat_baselines.tensorboard_dir="${TB_DIR}" \
  habitat_baselines.test_episode_count="${N_EP}" \
  habitat_baselines.eval.video_option='["disk"]'
