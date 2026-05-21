#!/usr/bin/env bash
# Run VLFM's own HabitatITMPolicyV2 (VLM-guided ObjectNav policy) on a single
# val episode, with habitat-baselines' eval.video_option=[disk] enabled so the
# 4-panel composite mp4 is written:
#
#     [annotated_rgb | annotated_depth | top_down_map]
#     [obstacle_map | value_map | (text overlay)]
#
# Composition is done by vlfm/utils/habitat_visualizer.py::collect_data, which
# pulls top_down_map from the frontier_exploration_map measurement (already in
# the default config) and obstacle_map / value_map from the policy's policy_info
# dict.  Last frame at episode end is written via habitat_baselines.generate_video.
#
# Prereqs (already running on this box - DON'T relaunch):
#   tmux session vlm_servers_7585, 4 panes:
#     pane 0  GroundingDINO  :12181  (GPU 0)
#     pane 1  BLIP2-ITM      :12182  (GPU 7)
#     pane 2  SAM            :12183  (GPU 0)
#     pane 3  YOLOv7         :12184  (GPU 0)
#   To start from scratch instead:
#     bash scripts/launch_vlm_servers_jy.sh 0
#
# GPU layout for THIS process:
#   CUDA_VISIBLE_DEVICES=4,5  -> cuda:0 = phys 4 (sim renderer),
#                                cuda:1 = phys 5 (torch policy actor)
#   Same split-GPU config that proved out in outputs/run_vlfm_split_gpu*.sh.
#   VLM servers talk over HTTP, they live on OTHER physical GPUs (0 / 7).
#
# Why this is the "real" video, not the per-step PNG dump:
#   - PNG dump (run_vlfm_split_gpu_frames.sh) only captures raw RGB from the
#     SIM tap.  No detections, no maps.  Good for proving the renderer is
#     alive, bad for showing what VLFM "sees".
#   - This script lets the habitat-baselines video writer run end-to-end and
#     produces a single mp4 with all 5 panels per frame.  Requires the
#     episode to actually reach done=True before generate_video() is called.
#
# Honest note on the prior "video errors out" claim:
#   We hit a timeout, NOT a crash, on the earlier OracleFBE + video_option=[disk]
#   run (see outputs/vlfm_split_gpu_video.log).  OracleFBE got stuck in a
#   collision loop at step ~22 and never reached done=True within 480s.  The
#   "core dumped" trailer was GNU `timeout` SIGKILLing the process, NOT a
#   Python crash.  HabitatITMPolicyV2 is the actual VLFM policy and should
#   reach done=True (success or step-cap stop) given enough wall time.

set -u

cd "$(dirname "$0")/.."
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_cuda_sim

# Two truly-empty GPUs for sim + torch.  Verified at 22:13 today:
#   GPU 1, 2, 4, 5 all at 4 MiB used; GPU 0 / 6 / 7 hosting other tenants.
# Re-use the same pair (4, 5) as the proven split-GPU baseline.
export CUDA_VISIBLE_DEVICES=4,5

# Probe taps so we can confirm the renderer is alive (>=6 distinct rgb_md5
# at the SIM tap).  Cheap, no behavior change.
# Also routes VLFM's internal PointNav sub-policy onto the same CUDA device
# as the main torch actor (cuda:1 here), via a monkey-patch of
# WrappedPointNavResNetPolicy.__init__ - no source change to vlfm/policy/.
# Needed because that wrapper defaults `device="cuda"` -> cuda:0, which
# conflicts with our split-GPU layout (torch_gpu_id=1).
export PYTHONPATH=outputs/probe_pkg
export VLFM_POINTNAV_GPU_ID=1

# Same ThreadedVectorEnv path that worked in the baseline.  Default forkserver
# VectorEnv has its own EGL-init worker hang on this box.
export HABITAT_ENV_DEBUG=1
export HF_ENDPOINT=https://hf-mirror.com

# VLM server ports (defaults match the running vlm_servers_7585 panes).
export GROUNDING_DINO_PORT=12181
export BLIP2ITM_PORT=12182
export SAM_PORT=12183
export YOLOV7_PORT=12184

VIDEO_DIR=video_dir/vlfm_itm_video
TB_DIR=tb/vlfm_itm_video
LOG=outputs/vlfm_itm_video.log
# Budget for ONE val episode: BLIP2 cosine() takes ~4s/step on this box,
# default max_episode_steps=500.  Allow 35 min (~2100s) so a full-length
# episode can finish + the mp4 flush has headroom.  Episodes usually end
# earlier on success or stop signal.
TIMEOUT_S=2100

mkdir -p "${VIDEO_DIR}" "${TB_DIR}"

echo "================================================================"
echo "vlfm.run, HabitatITMPolicyV2 + video_option=[disk]"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  (sim=cuda:0, torch=cuda:1)"
echo "VLM ports: GDINO=${GROUNDING_DINO_PORT} BLIP2=${BLIP2ITM_PORT} SAM=${SAM_PORT} YOLOv7=${YOLOV7_PORT}"
echo "video dir: ${VIDEO_DIR}/   tb dir: ${TB_DIR}/   log: ${LOG}"
echo "timeout: ${TIMEOUT_S}s"
echo "-- pre-flight GPU usage (sim+torch GPUs 4/5 must be ~4 MiB) --"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | head -10
echo "-- quick VLM health probes --"
for p in 12181 12182 12183 12184; do
  if curl -fsS -o /dev/null --max-time 2 "http://127.0.0.1:${p}/"; then
    echo "  port ${p}: up (HTTP responded)"
  else
    # 404 / 405 still counts as up - we just want to know the socket exists.
    code=$(curl -s -o /dev/null --max-time 2 -w '%{http_code}' "http://127.0.0.1:${p}/")
    if [ "${code}" != "000" ]; then
      echo "  port ${p}: up (HTTP ${code})"
    else
      echo "  port ${p}: DOWN - run scripts/launch_vlm_servers_jy.sh 0"
    fi
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
  habitat_baselines.eval.split=val \
  habitat_baselines.test_episode_count=1 \
  'habitat_baselines.eval.video_option=[disk]' \
  habitat.simulator.create_renderer=True \
  habitat.simulator.scene_dataset=data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json \
  habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
  habitat_baselines.torch_gpu_id=1 \
  habitat_baselines.video_dir="${VIDEO_DIR}" \
  habitat_baselines.tensorboard_dir="${TB_DIR}" \
  > "${LOG}" 2>&1
EXIT=$?

echo "EXIT=${EXIT}  (124 = trainer-side timeout, 0 = clean exit)"
echo "-- tap counts --"
rg -o '\[PROBE [^]]*tag=([A-Z_]+)' -r '$1' "${LOG}" 2>/dev/null | sort | uniq -c | sort -nr | head -10 || true
echo "-- distinct SIM (rgb,depth) tuples --"
N_DISTINCT=$(rg -F 'tag=SIM ' "${LOG}" 2>/dev/null | rg -oP 'rgb_md5=\w+ \| depth_md5=\w+' | sort -u | wc -l)
echo "${N_DISTINCT}"
echo "-- last 5 policy steps (Step / Mode / Action) --"
grep -E '^Step: [0-9]+ \|' "${LOG}" | tail -5 || true
echo "-- episode metrics --"
rg -iE 'success rate|spl|distance_to_goal|success: [01]|Failure cause' "${LOG}" | tail -10 || true
echo "-- video files written --"
ls -lh "${VIDEO_DIR}/" 2>/dev/null || true
echo "-- post-run GPU usage --"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | head -10
echo "DONE"
