#!/usr/bin/env bash
# Run the single cat-finding demo episode against the modified TEEsavR23oF scene.
#
# Pre-requisite: VLM servers must be running. If they're not:
#   bash scripts/launch_vlm_servers_jy.sh
#
# Split-GPU layout (required on multi-tenant boxes — co-locating habitat-sim's
# EGL renderer with torch's CUDA context on the same GPU triggers a
# renderer-freeze bug. See scripts/eval_1b_parallel.sh header for the same topology.
# Default: CUDA_VISIBLE_DEVICES=0,7  -> sim on cuda:0 (=GPU0), torch on cuda:1 (=GPU7).
# Override via env:
#   CUDA_VISIBLE_DEVICES=<sim_gpu>,<torch_gpu> bash scripts/cat_demo/eval_cat_demo.sh
#
# Output:
#   video_dir/cat_demo_<timestamp>/  -> rendered MP4 of the episode
#   tb/cat_demo_<timestamp>/         -> tensorboard metrics
set -u

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_pip

# Two GPUs: cuda:0 = sim renderer, cuda:1 = torch policy actor.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,7}"

# Auto-load the device-routing patch (puts VLFM's 3 hardcoded device="cuda"
# literals onto cuda:1 to match the main actor).
export PYTHONPATH="scripts/lib/vlfm_split_gpu_patch${PYTHONPATH:+:${PYTHONPATH}}"
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
POINTNAV_POLICY_PATH="${POINTNAV_POLICY_PATH:-data/pointnav_weights.pth}"
if [ ! -f "${POINTNAV_POLICY_PATH}" ] && [ -f "/data/jinsong.yuan/vlfm-demo/habitat-pointnav-demo/data/pointnav_weights.pth" ]; then
  POINTNAV_POLICY_PATH="/data/jinsong.yuan/vlfm-demo/habitat-pointnav-demo/data/pointnav_weights.pth"
fi
SUCCESS_DISTANCE="${SUCCESS_DISTANCE:-0.5}"

mkdir -p "${VIDEO_DIR}" "${TB_DIR}" "$(dirname "${LOG}")"

echo ">>> Cat-finding demo  (split-GPU)"
echo ">>> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  (sim=cuda:0, torch=cuda:1)"
echo ">>> scene=TEEsavR23oF (with cat merged into val/00800-...)"
echo ">>> split=${SPLIT}  N_EP=${N_EP}"
echo ">>> video_dir=${VIDEO_DIR}"
echo ">>> tb_dir   =${TB_DIR}"
echo ">>> log      =${LOG}"
echo ">>> pointnav =${POINTNAV_POLICY_PATH}"
echo ">>> success_distance=${SUCCESS_DISTANCE}"
echo

echo "-- episode pose --"
python - <<'PY' || true
import gzip
import json
import math
from pathlib import Path

episode = Path("data/datasets/objectnav/hm3d/v1/cat_demo/content/TEEsavR23oF.json.gz")
with gzip.open(episode, "rt") as f:
    data = json.load(f)
ep = data["episodes"][0]
rot = ep["start_rotation"]
yaw = math.degrees(2.0 * math.atan2(rot[1], rot[3]))
goals = next(iter(data["goals_by_category"].values()))
goal = goals[0]
print(f"  start_position={ep['start_position']}")
print(f"  start_yaw={yaw:+.2f} deg")
print(f"  target={goal['object_name']} position={goal['position']}")
print(f"  view_points={len(goal.get('view_points', []))}")
print(f"  info={ep.get('info', {})}")
PY
echo

echo "-- pre-flight GPU usage --"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | head -1
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tail -n +2 | awk -F',' -v visible="${CUDA_VISIBLE_DEVICES}" '
  BEGIN {
    split(visible, gpu_ids, ",")
    for (i in gpu_ids) {
      gsub(/ /, "", gpu_ids[i])
      want[gpu_ids[i]] = 1
    }
  }
  { gsub(/ /,""); idx=$1; if (idx in want) print "  GPU "$1": used="$2" free="$3 }'

echo "-- VLM health probes --"
health_ports=(12181 12182 12183 12184)
if [ "${VLFM_ATTR_VERIFY:-0}" != "0" ] && { [ -n "${VLFM_OBJECTNAV_QUERY:-}" ] || [ -n "${VLFM_ATTR_QUERY:-}" ] || [ -n "${VLFM_ATTR_PREDICATE:-}" ]; }; then
  health_ports+=("${ATTR_VERIFIER_PORT:-12186}")
fi
for port in "${health_ports[@]}"; do
  code=$(curl -s -o /dev/null --max-time 2 -w '%{http_code}' "http://127.0.0.1:${port}/" || true)
  if [ -n "${code}" ] && [ "${code}" != "000" ]; then
    echo "  port ${port}: up (HTTP ${code})"
  else
    echo "  port ${port}: DOWN -- run scripts/launch_vlm_servers_jy.sh 0"
  fi
done
echo

# Cat-on-furniture adjustments:
# - pointnav_stop_radius=1.3: the cat sits on a bed; the nearest navigable
#   point is ~1.0 m from the cat center, so the default 0.9 can never trigger
#   STOP and the episode runs out the step budget.
# - success_distance=0.5: goal view_points are navigable floor cells beside
#   the bed; the object-map projection of a cat on furniture is noisy, so the
#   default 0.1 m is not reachable in practice.
python -um vlfm.run \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=data/dummy_policy.pth \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.rl.policy.name=HabitatITMPolicyV2 \
  habitat_baselines.rl.policy.pointnav_policy_path="${POINTNAV_POLICY_PATH}" \
  habitat_baselines.rl.policy.pointnav_stop_radius=1.2 \
  habitat.task.measurements.success.success_distance="${SUCCESS_DISTANCE}" \
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
