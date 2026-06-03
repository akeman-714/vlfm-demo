#!/usr/bin/env bash
# Scenario 2: can TWO EGL render contexts share ONE physical GPU?
# Two independent vlfm.run workers, BOTH rendering on phys GPU1, both with
# nav on the VLM card (phys GPU0, proven-OK 1b routing). Different scenes,
# separate RGB-dump dirs. If both workers' frames advance -> EGL<->EGL
# coexistence works on Hopper -> 2 workers fit on 2 cards.
set -u
cd /data/jinsong.yuan/vlfm-demo/vlfm
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_cuda_sim
export PYTHONPATH=scripts/vlfm_split_gpu_patch
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export GROUNDING_DINO_PORT=12181 BLIP2ITM_PORT=12182 SAM_PORT=12183 YOLOV7_PORT=12184

run_worker() {  # $1=tag  $2=scene  $3=dumpdir
  local tag="$1" scene="$2" dump="$3"
  rm -rf "$dump"; mkdir -p "$dump"
  CUDA_VISIBLE_DEVICES=1,0 \
  VLFM_POINTNAV_GPU_ID=1 \
  HABITAT_ENV_DEBUG=1 \
  VLFM_DUMP_RGB_DIR="$dump" \
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
    "habitat.dataset.content_scenes=[$scene]" \
    habitat.simulator.create_renderer=True \
    habitat.simulator.scene_dataset=data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json \
    habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
    habitat_baselines.torch_gpu_id=1 > "/tmp/vlfm_s2_${tag}.log" 2>&1
  echo "[$tag] EXIT=$?"
}

run_worker A 4ok3usBNeis /tmp/vlfm_s2_A &
PIDA=$!
sleep 8
run_worker B 5cdEh9F2hJL /tmp/vlfm_s2_B &
PIDB=$!
echo "launched A=$PIDA B=$PIDB (both render on phys GPU1)"
wait "$PIDA" "$PIDB"
echo "BOTH_DONE"
