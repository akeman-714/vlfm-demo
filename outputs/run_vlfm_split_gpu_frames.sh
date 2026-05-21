#!/usr/bin/env bash
# Re-run the proven-good split-GPU trainer config (run_vlfm_split_gpu.sh),
# but ALSO dump every per-step RGB frame from the SIM tap to PNG so we can
# turn it into an mp4 with ffmpeg afterwards.
#
# Why this and not habitat_baselines.eval.video_option=[disk]?
#   Re-reading outputs/vlfm_split_gpu_video.log carefully: that run did NOT
#   crash - there is no Python traceback and no native SIGABRT.  The
#   "core dumped" line at the end is just GNU `timeout` SIGKILLing the
#   process when the 480s wall-clock budget ran out, while OracleFBE was
#   stuck in its action=1/collided=True loop (only 26 policy steps in 480s
#   under video_option=[disk], vs ~252 steps in 300s without it).
#   For OracleFBE specifically the disk-video machinery just adds enough
#   overhead that nothing ever reaches done=True before timeout, so no mp4
#   is flushed.  Per-step PNG dump (this script) avoids that whole
#   end-of-episode flush dependency and gives us a usable mp4 even when
#   the policy is buggy.
#
# Our probe-side approach is much simpler: outputs/probe_pkg/sitecustomize.py
# already taps Simulator.get_sensor_observations.  When PROBE_SAVE_FRAMES_DIR
# is set, every rgb frame is dropped to <dir>/frame_NNNNNN.png with PIL.
# No video encoder involved, no cv2/ffmpeg-during-trainer risk.
#
# Note about expected output:
#   The OracleFBE policy is buggy on this scene - it gets stuck against a
#   wall around step ~25 (action=MOVE_FORWARD, collided=True every tick,
#   pos doesn't change).  So the video will show: ~25 steps of genuine
#   exploration with frames varying, then a long tail of the same frame
#   repeating.  That tail is the OracleFBE bug, NOT a renderer freeze.
#   The point of recording is to *prove* the renderer is alive (frames 0-25
#   move smoothly) - not to demonstrate good navigation.
set -u

cd "$(dirname "$0")/.."
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_cuda_sim

# IMPORTANT: this trainer wants two GPUs that are TRULY EMPTY (a few MiB
# used).  On this multi-tenant H20-3e + driver 570 + EGL surfaceless box,
# habitat-sim's renderer cannot share a GPU with any other heavy CUDA
# workload - even bare sim (no torch) freezes to a constant garbage frame
# when another tenant is on the same GPU
# (proven by sim-only A_sim_step run on GPU 3 in /tmp/sim_a_gpu3.log:
#  >>> FROZEN, depth_uniq=1/5, avg 10.3s/step, no torch involved).
# nvidia-smi pre-flight showed GPUs 1, 2, 4, 5 all at ~4 MiB used.  Picking
# 4 + 5 to stay consistent with the original passing run.
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH=outputs/probe_pkg
export HF_ENDPOINT=https://hf-mirror.com
export HABITAT_ENV_DEBUG=1

# Frame dumping configuration (read by sitecustomize.py).
FRAMES_DIR=outputs/vlfm_split_gpu_frames
rm -rf "${FRAMES_DIR}"
mkdir -p "${FRAMES_DIR}"
export PROBE_SAVE_FRAMES_DIR="${FRAMES_DIR}"
export PROBE_SAVE_EVERY=2   # save every other frame; halves PNG write
                            # overhead so the trainer stays at ~1-2 s/step
                            # like the no-frames run did (302 steps in 360 s)

LOG=outputs/vlfm_split_gpu_frames.log
TIMEOUT_S=300   # ~250 steps at ~1.2s/step (with every-other-frame save);
                # enough that BOTH the genuine exploration phase (~25 steps
                # of varied frames) AND the OracleFBE wall-collision tail
                # are captured

echo "================================================================"
echo "vlfm.run, split-GPU + frame-by-frame PNG dump (no cv2/ffmpeg in-loop)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (sim=cuda:0 / gpu4, torch=cuda:1 / gpu5)"
echo "frames dir: ${FRAMES_DIR}/  (save_every=${PROBE_SAVE_EVERY})"
echo "log:        ${LOG}"
echo "timeout:    ${TIMEOUT_S}s"
echo "-- pre-flight GPU memory (sim+torch GPUs must be near-empty) --"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | head -10
echo "================================================================"

timeout "${TIMEOUT_S}" python -um vlfm.run \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=data/dummy_policy.pth \
  habitat_baselines.load_resume_state_config=False \
  habitat.task.lab_sensors.base_explorer.turn_angle=30 \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.split=val \
  habitat_baselines.test_episode_count=1 \
  'habitat_baselines.eval.video_option=[]' \
  habitat.simulator.create_renderer=True \
  habitat.simulator.scene_dataset=data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json \
  habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
  habitat_baselines.torch_gpu_id=1 \
  habitat_baselines.rl.policy.name=OracleFBEPolicy \
  habitat_baselines.video_dir=video_dir/vlfm_split_gpu_frames \
  habitat_baselines.tensorboard_dir=tb/vlfm_split_gpu_frames \
  > "${LOG}" 2>&1
EXIT=$?

echo "EXIT=${EXIT}  (124=timeout, 0=clean)"
N_FRAMES=$(ls "${FRAMES_DIR}"/frame_*.png 2>/dev/null | wc -l)
echo "frames dumped: ${N_FRAMES}"

N_DISTINCT=$(rg -F 'tag=SIM ' "${LOG}" 2>/dev/null | rg -oP 'rgb_md5=\w+ \| depth_md5=\w+' | sort -u | wc -l)
echo "distinct (rgb,depth) tuples seen at SIM tap: ${N_DISTINCT}"
echo "-- post-run GPU memory --"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | head -10

if [ "${N_FRAMES}" -gt 0 ] && [ "${N_DISTINCT}" -gt 2 ]; then
  echo "================================================================"
  echo "Encoding mp4 with ffmpeg @ 8fps..."
  echo "================================================================"
  MP4="${FRAMES_DIR}/vlfm_split_gpu.mp4"
  ffmpeg -y -loglevel warning \
      -framerate 8 \
      -pattern_type glob -i "${FRAMES_DIR}/frame_*.png" \
      -c:v libx264 -pix_fmt yuv420p -crf 23 \
      -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
      "${MP4}" 2>&1 | tail -20
  ls -lh "${MP4}" 2>/dev/null || echo "ffmpeg failed; PNGs are still in ${FRAMES_DIR}/"
  echo "MP4 path: ${MP4}"
elif [ "${N_DISTINCT}" -le 2 ]; then
  echo "WARNING: only ${N_DISTINCT} distinct frames seen - renderer was frozen."
  echo "         Most likely cause: another tenant grabbed the sim GPU mid-run."
  echo "         Check 'post-run GPU memory' line above vs the pre-flight line."
fi

echo "DONE"
