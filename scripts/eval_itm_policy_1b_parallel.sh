#!/usr/bin/env bash
# 1b parallel ObjectNav eval (the validated "render-card-CUDA-free" topology).
#
# N parallel workers: each habitat renderer (EGL) runs ALONE on its own GPU,
# while ALL torch/nav contexts + the 4 VLM Flask servers share ONE card
# (VLM_NAV_CARD). Verified on H20: the render card MUST stay CUDA-free or the
# first-person view freezes; nav co-locates fine with the VLMs. Two EGL
# renderers CANNOT share a card. So the GPU floor is N+1 (N render cards + 1
# shared VLM&nav card).
#
# Pre-req: VLM servers already running on $VLM_NAV_CARD:
#   bash scripts/launch_vlm_servers_jy.sh $VLM_NAV_CARD
#
# Env vars (all optional; defaults reproduce the 5-card / 4-parallel run):
#   RENDER_CARDS   comma list of physical GPU ids, one worker each  (default "1,2,4,5")
#   VLM_NAV_CARD   physical GPU id hosting VLMs + every nav          (default 0)
#   EP_PER_WORKER  episodes per worker                               (default 3)
#   SCENES         comma list of HM3D scenes, one per render card    (default 4 val scenes)
#   SPLIT          dataset split                                     (default val)
#   VIDEO_DIR      output root for per-worker mp4s + logs            (default /tmp/vlfm_1b_video)
#   TIMEOUT_S      per-worker wall budget [s]                        (default 1800)
#   SAMPLE_GPU     1 = sample VLM_NAV_CARD util every 1s + print min/median/p90/max (default 0)
set -u
cd "$(dirname "$0")/.."
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_cuda_sim
export PYTHONPATH="scripts/vlfm_split_gpu_patch${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-12181}"
export BLIP2ITM_PORT="${BLIP2ITM_PORT:-12182}"
export SAM_PORT="${SAM_PORT:-12183}"
export YOLOV7_PORT="${YOLOV7_PORT:-12184}"

RENDER_CARDS="${RENDER_CARDS:-1,2,4,5}"
VLM_NAV_CARD="${VLM_NAV_CARD:-0}"
EP_PER_WORKER="${EP_PER_WORKER:-3}"
SCENES="${SCENES:-4ok3usBNeis,5cdEh9F2hJL,6s7QHgap2fW,bxsVRursffK}"
SPLIT="${SPLIT:-val}"
VIDEO_DIR="${VIDEO_DIR:-/tmp/vlfm_1b_video}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
SAMPLE_GPU="${SAMPLE_GPU:-0}"

IFS=',' read -ra RC <<<"$RENDER_CARDS"
IFS=',' read -ra SC <<<"$SCENES"
rm -rf "$VIDEO_DIR"; mkdir -p "$VIDEO_DIR"

echo "=============================================================="
echo "1b parallel eval: ${#RC[@]} workers"
echo "  render cards : $RENDER_CARDS  (one EGL renderer each, CUDA-free)"
echo "  VLM+nav card : $VLM_NAV_CARD  (4 VLM servers + all navs)"
echo "  ep/worker=$EP_PER_WORKER  split=$SPLIT  timeout=${TIMEOUT_S}s  sample_gpu=$SAMPLE_GPU"
echo "  video_dir=$VIDEO_DIR"
echo "-- VLM health --"
for p in "$GROUNDING_DINO_PORT" "$BLIP2ITM_PORT" "$SAM_PORT" "$YOLOV7_PORT"; do
  echo "  port $p: HTTP $(curl -s -o /dev/null --max-time 2 -w '%{http_code}' http://127.0.0.1:$p/ || echo down)"
done
echo "=============================================================="

UTIL="$VIDEO_DIR/gpu${VLM_NAV_CARD}_util.log"
SAMPLER=""
if [ "$SAMPLE_GPU" = "1" ]; then
  ( while true; do echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i "$VLM_NAV_CARD")"; sleep 1; done ) > "$UTIL" 2>&1 &
  SAMPLER=$!
fi

run_worker() {  # $1=tag $2=render_card $3=scene
  local tag="$1" rc="$2" scene="$3"; local vdir="$VIDEO_DIR/$tag"; mkdir -p "$vdir"
  CUDA_VISIBLE_DEVICES="${rc},${VLM_NAV_CARD}" \
  VLFM_POINTNAV_GPU_ID=1 \
  HABITAT_ENV_DEBUG=1 \
  timeout "$TIMEOUT_S" python -um vlfm.run \
    habitat_baselines.evaluate=True \
    habitat_baselines.eval_ckpt_path_dir=data/dummy_policy.pth \
    habitat_baselines.load_resume_state_config=False \
    habitat_baselines.rl.policy.name=HabitatITMPolicyV2 \
    habitat.task.lab_sensors.base_explorer.turn_angle=30 \
    habitat_baselines.num_environments=1 \
    habitat_baselines.eval.split="$SPLIT" \
    habitat_baselines.test_episode_count="$EP_PER_WORKER" \
    "habitat_baselines.eval.video_option=[disk]" \
    habitat_baselines.video_dir="$vdir" \
    habitat.dataset.data_path="data/datasets/objectnav/hm3d/v1/${SPLIT}/${SPLIT}.json.gz" \
    "habitat.dataset.content_scenes=[$scene]" \
    habitat.simulator.create_renderer=True \
    habitat.simulator.scene_dataset=data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json \
    habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
    habitat_baselines.torch_gpu_id=1 > "$vdir/run.log" 2>&1
  echo "[$tag] EXIT=$? card=$rc scene=$scene videos=$(ls "$vdir"/*.mp4 2>/dev/null | wc -l)"
}

PIDS=()
for idx in "${!RC[@]}"; do
  tag=$(printf "w%d_gpu%s" "$idx" "${RC[$idx]}")
  scene="${SC[$((idx % ${#SC[@]}))]}"
  run_worker "$tag" "${RC[$idx]}" "$scene" &
  PIDS+=($!)
  sleep 6
done
echo "launched ${#PIDS[@]} workers: ${PIDS[*]}"
wait "${PIDS[@]}"
[ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null
echo "ALL_DONE"

if [ "$SAMPLE_GPU" = "1" ]; then
  echo "=== GPU${VLM_NAV_CARD} util (VLM+nav card) ==="
  awk '{gsub(/%|,/,""); print $2}' "$UTIL" | sort -n | awk '{a[NR]=$1} END{if(NR){print "samples="NR; print "min="a[1]"%"; print "median="a[int(NR/2)]"%"; print "p90="a[int(NR*0.9)]"%"; print "max="a[NR]"%"}}'
fi
echo "=== results (success = filename contains success=1.00) ==="
for d in "$VIDEO_DIR"/*/; do
  [ -d "$d" ] || continue
  t=$(basename "$d"); n=$(ls "$d"/*.mp4 2>/dev/null | wc -l); ok=$(ls "$d"/*success=1.00*.mp4 2>/dev/null | wc -l)
  echo "$t: $n videos, $ok success"
done
