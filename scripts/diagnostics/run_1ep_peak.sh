#!/usr/bin/env bash
# Measure GPU0 peak under the real workload: 4 parallel workers (render on phys
# GPU 1/2/4/5), VLM + 4 navs on GPU0, 1 full episode each, video on.
# A 1s sampler logs GPU0 util+mem for the whole run.
set -u
cd /data/jinsong.yuan/vlfm-demo/vlfm
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_cuda_sim
export PYTHONPATH=scripts/vlfm_split_gpu_patch
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export GROUNDING_DINO_PORT=12181 BLIP2ITM_PORT=12182 SAM_PORT=12183 YOLOV7_PORT=12184

VROOT=/tmp/vlfm_1ep_video; rm -rf "$VROOT"; mkdir -p "$VROOT"
UTIL=/tmp/vlfm_1ep_gpu0_util.log; : > "$UTIL"

# 1s GPU0 sampler.
( while true; do echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 0)"; sleep 1; done ) > "$UTIL" 2>&1 &
SAMPLER=$!

run_worker() {  # $1=tag $2=render_card $3=scene
  local tag="$1" rc="$2" scene="$3"; local vdir="$VROOT/$tag"; mkdir -p "$vdir"
  CUDA_VISIBLE_DEVICES="${rc},0" VLFM_POINTNAV_GPU_ID=1 HABITAT_ENV_DEBUG=1 \
  timeout 1200 python -um vlfm.run \
    habitat_baselines.evaluate=True \
    habitat_baselines.eval_ckpt_path_dir=data/dummy_policy.pth \
    habitat_baselines.load_resume_state_config=False \
    habitat_baselines.rl.policy.name=HabitatITMPolicyV2 \
    habitat.task.lab_sensors.base_explorer.turn_angle=30 \
    habitat_baselines.num_environments=1 \
    habitat_baselines.eval.split=val \
    habitat_baselines.test_episode_count=1 \
    "habitat_baselines.eval.video_option=[disk]" \
    habitat_baselines.video_dir="$vdir" \
    habitat.dataset.data_path=data/datasets/objectnav/hm3d/v1/val/val.json.gz \
    "habitat.dataset.content_scenes=[$scene]" \
    habitat.simulator.create_renderer=True \
    habitat.simulator.scene_dataset=data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json \
    habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
    habitat_baselines.torch_gpu_id=1 > "/tmp/vlfm_1ep_${tag}.log" 2>&1
  echo "[$tag] EXIT=$? videos=$(ls "$vdir"/*.mp4 2>/dev/null | wc -l)"
}

run_worker A 1 4ok3usBNeis & PA=$!; sleep 6
run_worker B 2 5cdEh9F2hJL & PB=$!; sleep 6
run_worker C 4 6s7QHgap2fW & PC=$!; sleep 6
run_worker D 5 bxsVRursffK & PD=$!
echo "launched A=$PA B=$PB C=$PC D=$PD"
wait "$PA" "$PB" "$PC" "$PD"
kill "$SAMPLER" 2>/dev/null
echo "ALL_DONE"
echo "=== GPU0 util stats ==="
awk '{gsub(/%|,/,""); print $2}' "$UTIL" | sort -n | awk '{a[NR]=$1} END{print "samples="NR; print "min="a[1]"%"; print "median="a[int(NR/2)]"%"; print "p90="a[int(NR*0.9)]"%"; print "max="a[NR]"%"}'
echo "mem_peak_MiB=$(awk '{print $4}' "$UTIL" | sort -n | tail -1)"
