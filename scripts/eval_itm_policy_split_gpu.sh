#!/usr/bin/env bash
# Evaluate VLFM's HabitatITMPolicyV2 on HM3D val with eval.video_option=[disk]
# so a full 5-panel composite mp4 is written (annotated RGB + annotated depth
# + top_down_map + obstacle_map + value_map + text overlay).  Sibling of the
# upstream scripts/eval_itm_policy.sh, with two additions:
#
#   1. Split-GPU layout: sim renderer on cuda:0, torch policy on cuda:1.
#      Required on multi-tenant boxes where co-locating habitat-sim's EGL
#      renderer and torch's CUDA context on the same GPU triggers a
#      renderer-freeze bug.
#
#   2. Auto-loaded sitecustomize patch under scripts/vlfm_split_gpu_patch/
#      that routes VLFM's three hardcoded device="cuda" literals to
#      cuda:${VLFM_POINTNAV_GPU_ID}.  No edits to vlfm/policy/ source.
#
# Prereqs:
#   - VLM servers running (GroundingDINO :12181, BLIP2-ITM :12182, SAM :12183,
#     YOLOv7 :12184).  To launch:  bash scripts/launch_vlm_servers_jy.sh 0
#   - conda env vlfm_cuda_sim activated by this script.
#   - At least two empty CUDA devices.  Default picks 4,5; override with
#     CUDA_VISIBLE_DEVICES=<sim>,<torch> in the caller's env.

set -u

cd "$(dirname "$0")/.."
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_cuda_sim

# Two GPUs: cuda:0 = sim renderer, cuda:1 = torch policy actor.
# Override by exporting CUDA_VISIBLE_DEVICES before invoking this script.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

# Auto-load the device-routing patch via Python's sitecustomize hook.
# VLFM_POINTNAV_GPU_ID must match habitat_baselines.torch_gpu_id below.
export PYTHONPATH="scripts/vlfm_split_gpu_patch${PYTHONPATH:+:${PYTHONPATH}}"
export VLFM_POINTNAV_GPU_ID="${VLFM_POINTNAV_GPU_ID:-1}"

# ThreadedVectorEnv path (sim and torch in the same process, on separate
# GPUs).  The default forkserver VectorEnv has an EGL-init worker hang on
# this box; the split-GPU layout is what lets us stay in-process safely.
export HABITAT_ENV_DEBUG=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-12181}"
export BLIP2ITM_PORT="${BLIP2ITM_PORT:-12182}"
export SAM_PORT="${SAM_PORT:-12183}"
export YOLOV7_PORT="${YOLOV7_PORT:-12184}"

VIDEO_DIR="${VIDEO_DIR:-video_dir/vlfm_itm_split_gpu}"
TB_DIR="${TB_DIR:-tb/vlfm_itm_split_gpu}"
LOG="${LOG:-outputs/vlfm_itm_split_gpu.log}"
# BLIP2 cosine() ~4 s/step on this box, max_episode_steps=500 -> ~35 min
# wall budget per episode plus headroom for the mp4 flush.
TIMEOUT_S="${TIMEOUT_S:-2100}"
N_EPISODES="${N_EPISODES:-1}"
SPLIT="${SPLIT:-val}"

mkdir -p "${VIDEO_DIR}" "${TB_DIR}" "$(dirname "${LOG}")"

echo "================================================================"
echo "vlfm.run, HabitatITMPolicyV2 + video_option=[disk]"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  (sim=cuda:0, torch=cuda:1)"
echo "VLM ports: GDINO=${GROUNDING_DINO_PORT} BLIP2=${BLIP2ITM_PORT} SAM=${SAM_PORT} YOLOv7=${YOLOV7_PORT}"
echo "video dir: ${VIDEO_DIR}/   tb dir: ${TB_DIR}/   log: ${LOG}"
echo "timeout: ${TIMEOUT_S}s   episodes: ${N_EPISODES}   split: ${SPLIT}"
echo "-- pre-flight GPU usage (sim+torch GPUs must be near-empty) --"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | head -10
echo "-- VLM health probes --"
for p in "${GROUNDING_DINO_PORT}" "${BLIP2ITM_PORT}" "${SAM_PORT}" "${YOLOV7_PORT}"; do
  code=$(curl -s -o /dev/null --max-time 2 -w '%{http_code}' "http://127.0.0.1:${p}/" || true)
  if [ "${code}" != "000" ] && [ -n "${code}" ]; then
    echo "  port ${p}: up (HTTP ${code})"
  else
    echo "  port ${p}: DOWN -- run scripts/launch_vlm_servers_jy.sh 0"
  fi
done
echo "================================================================"

timeout "${TIMEOUT_S}" python -um vlfm.run \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=data/dummy_policy.pth \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.rl.policy.name=HabitatITMPolicyV2 \
  habitat.task.lab_sensors.base_explorer.turn_angle=30 \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.split="${SPLIT}" \
  habitat_baselines.test_episode_count="${N_EPISODES}" \
  'habitat_baselines.eval.video_option=[disk]' \
  habitat.simulator.create_renderer=True \
  habitat.simulator.scene_dataset=data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json \
  habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
  habitat_baselines.torch_gpu_id="${VLFM_POINTNAV_GPU_ID}" \
  habitat_baselines.video_dir="${VIDEO_DIR}" \
  habitat_baselines.tensorboard_dir="${TB_DIR}" \
  > "${LOG}" 2>&1
EXIT=$?

echo "EXIT=${EXIT}  (124 = wall-clock timeout, 0 = clean exit)"
echo "-- episode metrics --"
rg -iE 'success rate|spl|distance_to_goal|success: [01]|Failure cause' "${LOG}" | tail -10 || true
echo "-- video files written --"
ls -lh "${VIDEO_DIR}/" 2>/dev/null || true
echo "-- post-run GPU usage --"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | head -10
echo "DONE"
