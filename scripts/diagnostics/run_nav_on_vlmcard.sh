#!/usr/bin/env bash
# Test the "nav off the render card" hypothesis:
#   render/EGL alone on a clean card (GPU1), nav/torch routed onto the VLM card
#   (GPU0, where the 4 VLM servers already live). If the render card is truly
#   CUDA-free, the first-person view should NOT freeze -> unlocks the 3-card
#   parallel topology (1 shared VLM+nav card + 1 sim card per worker).
#
# CUDA_VISIBLE_DEVICES=1,0  => visible idx0=phys GPU1 (render), idx1=phys GPU0 (nav+VLM)
set -u
cd /data/jinsong.yuan/vlfm-demo/vlfm
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_cuda_sim
export CUDA_VISIBLE_DEVICES=1,0
export PYTHONPATH=scripts/vlfm_split_gpu_patch
export VLFM_POINTNAV_GPU_ID=1            # torch/nav -> visible idx1 = phys GPU0 (VLM card)
export HABITAT_ENV_DEBUG=1               # ThreadedVectorEnv (same as working split)
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export GROUNDING_DINO_PORT=12181 BLIP2ITM_PORT=12182 SAM_PORT=12183 YOLOV7_PORT=12184

export VLFM_DUMP_RGB_DIR=/tmp/vlfm_rgbdump_navvlm
rm -rf "$VLFM_DUMP_RGB_DIR"; mkdir -p "$VLFM_DUMP_RGB_DIR"

timeout 300 python -um vlfm.run \
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
