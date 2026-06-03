#!/usr/bin/env bash
set -u
cd /data/jinsong.yuan/vlfm-demo/vlfm
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_cuda_sim
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=scripts/vlfm_split_gpu_patch
export VLFM_POINTNAV_GPU_ID=0
export HABITAT_ENV_DEBUG=1
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export GROUNDING_DINO_PORT=12181 BLIP2ITM_PORT=12182 SAM_PORT=12183 YOLOV7_PORT=12184

export VLFM_DUMP_RGB_DIR=/tmp/vlfm_rgbdump
rm -rf "$VLFM_DUMP_RGB_DIR"; mkdir -p "$VLFM_DUMP_RGB_DIR"

timeout 900 python -um vlfm.run \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=data/dummy_policy.pth \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.rl.policy.name=HabitatITMPolicyV2 \
  habitat.task.lab_sensors.base_explorer.turn_angle=30 \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.split=val \
  habitat_baselines.test_episode_count=1 \
  "habitat_baselines.eval.video_option=[disk]" \
  habitat_baselines.video_dir=/tmp/vlfm_dump_video \
  habitat.dataset.data_path=data/datasets/objectnav/hm3d/v1/val/val.json.gz \
  "habitat.dataset.content_scenes=[4ok3usBNeis]" \
  habitat.simulator.create_renderer=True \
  habitat.simulator.scene_dataset=data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json \
  habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
  habitat_baselines.torch_gpu_id=0
echo "EXIT_CODE=$?"
