#!/usr/bin/env bash
# Run the single cat-finding demo episode against the modified TEEsavR23oF scene.
#
# Pre-requisite: VLM servers must be running. If they're not:
#   bash scripts/launch_vlm_servers_jy.sh
#
# Split-GPU layout (required on multi-tenant boxes — co-locating habitat-sim's
# EGL renderer with torch's CUDA context on the same GPU triggers a
# renderer-freeze bug; see scripts/eval_itm_policy_split_gpu.sh header).
# Default: CUDA_VISIBLE_DEVICES=0,7  -> sim on cuda:0 (=GPU0), torch on cuda:1 (=GPU7).
# Override via env:
#   CUDA_VISIBLE_DEVICES=<sim_gpu>,<torch_gpu> bash scripts/eval_cat_demo.sh
#
# Output:
#   video_dir/cat_demo_<timestamp>/  -> rendered MP4 of the episode
#   tb/cat_demo_<timestamp>/         -> tensorboard metrics
set -u

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_pip

# Two GPUs: cuda:0 = sim renderer, cuda:1 = torch policy actor.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,7}"

# Auto-load the device-routing patch (puts VLFM's 3 hardcoded device="cuda"
# literals onto cuda:1 to match the main actor).
export PYTHONPATH="scripts/vlfm_split_gpu_patch${PYTHONPATH:+:${PYTHONPATH}}"
export VLFM_POINTNAV_GPU_ID="${VLFM_POINTNAV_GPU_ID:-1}"

# ThreadedVectorEnv (the forkserver path EGL-init-hangs on this box).
export HABITAT_ENV_DEBUG=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SPLIT="${SPLIT:-cat_demo}"
N_EP="${N_EP:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
VIDEO_DIR="${VIDEO_DIR:-video_dir/cat_demo_${TS}}"
TB_DIR="${TB_DIR:-tb/cat_demo_${TS}}"
LOG="${LOG:-outputs/cat_demo_${TS}.log}"

mkdir -p "${VIDEO_DIR}" "${TB_DIR}" "$(dirname "${LOG}")"

echo ">>> Cat-finding demo  (split-GPU)"
echo ">>> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  (sim=cuda:0, torch=cuda:1)"
echo ">>> scene=TEEsavR23oF (with cat merged into val/00800-...)"
echo ">>> split=${SPLIT}  N_EP=${N_EP}"
echo ">>> video_dir=${VIDEO_DIR}"
echo ">>> tb_dir   =${TB_DIR}"
echo ">>> log      =${LOG}"
echo

echo "-- pre-flight GPU usage --"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | head -1
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tail -n +2 | awk -F',' '
  { gsub(/ /,""); idx=$1; if (idx==0 || idx==7) print "  GPU "$1": used="$2" free="$3 }'

echo "-- VLM health probes --"
for port in 12181 12182 12183 12184; do
  code=$(curl -s -o /dev/null --max-time 2 -w '%{http_code}' "http://127.0.0.1:${port}/" || true)
  if [ -n "${code}" ] && [ "${code}" != "000" ]; then
    echo "  port ${port}: up (HTTP ${code})"
  else
    echo "  port ${port}: DOWN -- run scripts/launch_vlm_servers_jy.sh 0"
  fi
done
echo

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
  habitat_baselines.eval.video_option='["disk"]' \
  habitat.simulator.create_renderer=True \
  habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
  habitat_baselines.torch_gpu_id="${VLFM_POINTNAV_GPU_ID}" \
  > "${LOG}" 2>&1
EXIT=$?

echo "EXIT=${EXIT}   (0=clean, 124=timeout, 134=SIGABRT, 139=SIGSEGV)"
echo "-- tail of log --"
tail -n 30 "${LOG}" || true
echo "-- video files --"
ls -lh "${VIDEO_DIR}/" 2>/dev/null || true
exit "${EXIT}"
