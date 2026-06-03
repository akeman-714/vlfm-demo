#!/usr/bin/env bash
# Reproduce the "working" dual-card split-GPU layout WITH per-step RGB dump,
# same scene/data_path as the single-card tests, to check if the first-person
# view actually advances on dual-card or was only ever judged from logs.
set -u
cd /data/jinsong.yuan/vlfm-demo/vlfm
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_cuda_sim
export CUDA_VISIBLE_DEVICES=1,2          # sim=cuda:0 (phys1), torch=cuda:1 (phys2)
export PYTHONPATH=scripts/vlfm_split_gpu_patch
export VLFM_POINTNAV_GPU_ID=1
export HABITAT_ENV_DEBUG=1               # ThreadedVectorEnv, in-process, split across 2 GPUs
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export GROUNDING_DINO_PORT=12181 BLIP2ITM_PORT=12182 SAM_PORT=12183 YOLOV7_PORT=12184

export VLFM_DUMP_RGB_DIR=/tmp/vlfm_rgbdump_dual
rm -rf "$VLFM_DUMP_RGB_DIR"; mkdir -p "$VLFM_DUMP_RGB_DIR"

timeout 420 python -um vlfm.run \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=data/dummy_policy.pth \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.rl.policy.name=HabitatITMPolicyV2 \
  habitat.task.lab_sensors.base_explorer.turn_angle=30 \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.split=val \
  habitat_baselines.test_episode_count=1 \
  habitat.dataset.data_path=data/datasets/objectnav/hm3d/v1/val/val.json.gz \
  "habitat.dataset.content_scenes=[4ok3usBNeis]" \
  habitat.simulator.create_renderer=True \
  habitat.simulator.scene_dataset=data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json \
  habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
  habitat_baselines.torch_gpu_id=1
echo "EXIT_CODE=$?"
