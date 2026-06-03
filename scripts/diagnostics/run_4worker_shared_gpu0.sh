#!/usr/bin/env bash
# Stress test the 1b topology at N=4: four workers each render on their OWN card
# (phys GPU 1/2/4/5), while ALL nav/torch + the 4 VLM servers share ONE card (GPU0).
# Question: does GPU0 (shared CUDA card) become the bottleneck?
# Measure: per-worker step cadence (frame mtimes) + per-worker render health
# (unique md5) + GPU0 utilization samples.
set -u
cd /data/jinsong.yuan/vlfm-demo/vlfm
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_cuda_sim
export PYTHONPATH=scripts/vlfm_split_gpu_patch
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export GROUNDING_DINO_PORT=12181 BLIP2ITM_PORT=12182 SAM_PORT=12183 YOLOV7_PORT=12184

# GPU0 utilization sampler (every 2s) for the duration.
( for i in $(seq 1 180); do
    echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 0)";
    sleep 2;
  done ) > /tmp/vlfm_4w_gpu0_util.log 2>&1 &
SAMPLER=$!

run_worker() {  # $1=tag $2=render_card $3=scene $4=dumpdir
  local tag="$1" rc="$2" scene="$3" dump="$4"
  rm -rf "$dump"; mkdir -p "$dump"
  CUDA_VISIBLE_DEVICES="${rc},0" \
  VLFM_POINTNAV_GPU_ID=1 \
  HABITAT_ENV_DEBUG=1 \
  VLFM_DUMP_RGB_DIR="$dump" \
  timeout 340 python -um vlfm.run \
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
    habitat_baselines.torch_gpu_id=1 > "/tmp/vlfm_4w_${tag}.log" 2>&1
  echo "[$tag] EXIT=$?"
}

run_worker A 1 4ok3usBNeis /tmp/vlfm_4w_A & PA=$!; sleep 6
run_worker B 2 5cdEh9F2hJL /tmp/vlfm_4w_B & PB=$!; sleep 6
run_worker C 4 6s7QHgap2fW /tmp/vlfm_4w_C & PC=$!; sleep 6
run_worker D 5 bxsVRursffK /tmp/vlfm_4w_D & PD=$!
echo "launched A=$PA B=$PB C=$PC D=$PD"
wait "$PA" "$PB" "$PC" "$PD"
kill "$SAMPLER" 2>/dev/null
echo "ALL_DONE"
