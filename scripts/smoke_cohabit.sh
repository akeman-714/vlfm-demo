#!/usr/bin/env bash
# Smoke test: run 3 ObjectNav HM3D val episodes with sim + torch policy +
# PointNav sub-policy all CO-HABITING on the same physical GPU (the project's
# split-GPU script comment warns this triggers an "EGL renderer-freeze" bug;
# this smoke checks whether the claim still holds on H20-3e + driver 570).
#
# What this answers:
#   * Does sim+torch on one card freeze the renderer on this hardware?
#   * What's the actual per-episode wall time at cruising state?
#   * Does the fail-only video patch correctly skip success episodes?
#
# Independent variables (vs eval_itm_policy_split_gpu.sh):
#   * CUDA_VISIBLE_DEVICES=4         (one card instead of two)
#   * VLFM_POINTNAV_GPU_ID=0         (torch policy on same cuda:0 as sim)
#   * VLFM_SKIP_SUCCESS_VIDEOS=1     (exercise the new patch)
#   * N_EPISODES=3                   (smoke, not full run)
#   * TIMEOUT_S=900                  (3 ep budget, leaves room for slow ones)
#
# Re-uses GPU 0's existing VLM stack (ports 12181-12184).

set -u

cd "$(dirname "$0")/.."
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_cuda_sim

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONPATH="scripts/vlfm_split_gpu_patch${PYTHONPATH:+:${PYTHONPATH}}"
export VLFM_POINTNAV_GPU_ID="${VLFM_POINTNAV_GPU_ID:-0}"
export VLFM_SKIP_SUCCESS_VIDEOS="${VLFM_SKIP_SUCCESS_VIDEOS:-1}"
export HABITAT_ENV_DEBUG=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-12181}"
export BLIP2ITM_PORT="${BLIP2ITM_PORT:-12182}"
export SAM_PORT="${SAM_PORT:-12183}"
export YOLOV7_PORT="${YOLOV7_PORT:-12184}"

VIDEO_DIR="${VIDEO_DIR:-video_dir/smoke_cohabit_$(date +%Y%m%d_%H%M%S)}"
TB_DIR="${TB_DIR:-tb/smoke_cohabit_$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-outputs/smoke_cohabit_$(date +%Y%m%d_%H%M%S).log}"
TIMEOUT_S="${TIMEOUT_S:-900}"
N_EPISODES="${N_EPISODES:-3}"
SPLIT="${SPLIT:-val}"

mkdir -p "${VIDEO_DIR}" "${TB_DIR}" "$(dirname "${LOG}")"

echo "================================================================"
echo "smoke: vlfm.run HabitatITMPolicyV2, sim+torch COHABIT on one card"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  (sim=cuda:0, torch=cuda:0)"
echo "VLFM_SKIP_SUCCESS_VIDEOS=${VLFM_SKIP_SUCCESS_VIDEOS}"
echo "VLM ports: GDINO=${GROUNDING_DINO_PORT} BLIP2=${BLIP2ITM_PORT} SAM=${SAM_PORT} YOLOv7=${YOLOV7_PORT}"
echo "video dir: ${VIDEO_DIR}/   tb dir: ${TB_DIR}/   log: ${LOG}"
echo "timeout: ${TIMEOUT_S}s   episodes: ${N_EPISODES}   split: ${SPLIT}"
echo "-- pre-flight GPU usage --"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | head -10
echo "-- VLM health probes --"
for p in "${GROUNDING_DINO_PORT}" "${BLIP2ITM_PORT}" "${SAM_PORT}" "${YOLOV7_PORT}"; do
  code=$(curl -s -o /dev/null --max-time 2 -w '%{http_code}' "http://127.0.0.1:${p}/" || true)
  if [ "${code}" != "000" ] && [ -n "${code}" ]; then
    echo "  port ${p}: up (HTTP ${code})"
  else
    echo "  port ${p}: DOWN"
  fi
done
echo "================================================================"

T_START=$(date +%s)
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
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

echo "================================================================"
echo "EXIT=${EXIT}  (124 = wall-clock timeout, 0 = clean exit)"
echo "wall: ${ELAPSED}s  (~$((ELAPSED / 60))min)"
if [ "${N_EPISODES}" -gt 0 ] && [ "${ELAPSED}" -gt 0 ]; then
  echo "wall per episode: $((ELAPSED / N_EPISODES))s  (rough, includes warmup)"
fi
echo "-- patch boot trace (first 6 lines) --"
rg '\[VLFM_SPLIT_GPU_PATCH' "${LOG}" | head -6
echo "-- last 'Step:' lines per mode (sanity sim+policy ran) --"
rg 'Step:' "${LOG}" | tail -5
echo "-- episode boundaries --"
rg -i 'episode|stopiteration|distance_to_goal|success|spl|stop_called|target_detected' "${LOG}" | tail -20
echo "-- video files written (failures only thanks to patch) --"
ls -lh "${VIDEO_DIR}/" 2>/dev/null
echo "-- post-run GPU usage --"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | head -10
echo "DONE log=${LOG}"
