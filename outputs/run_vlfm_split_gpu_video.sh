#!/usr/bin/env bash
# Same as run_vlfm_split_gpu.sh but:
#   1) record per-episode video to disk (video_option=[disk])
#   2) shrink max_episode_steps so the episode actually reaches done=True
#      within our timeout (habitat-baselines only writes the mp4 after
#      episode termination - if we hit wall-clock TIMEOUT first, no video).
#
# The previous run achieved ~1.2 s/step (302 steps in ~360 s).  Cap
# max_episode_steps at 200 so episode reaches done in ~250 s, leaves ~110 s
# headroom for video encoding + cleanup.
set -u

cd "$(dirname "$0")/.."
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_cuda_sim

export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH=outputs/probe_pkg
export HF_ENDPOINT=https://hf-mirror.com
export HABITAT_ENV_DEBUG=1

LOG=outputs/vlfm_split_gpu_video.log
VIDEO_DIR=video_dir/vlfm_split_gpu_video
TIMEOUT_S=480
MAX_STEPS=200

mkdir -p "${VIDEO_DIR}"

echo "================================================================"
echo "vlfm.run, split-GPU + video recording"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (sim=cuda:0 / gpu4, torch=cuda:1 / gpu5)"
echo "max_episode_steps=${MAX_STEPS}  timeout=${TIMEOUT_S}s"
echo "video dir: ${VIDEO_DIR}/"
echo "log:       ${LOG}"
echo "================================================================"

timeout "${TIMEOUT_S}" python -um vlfm.run \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=data/dummy_policy.pth \
  habitat_baselines.load_resume_state_config=False \
  habitat.task.lab_sensors.base_explorer.turn_angle=30 \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.split=val \
  habitat_baselines.test_episode_count=1 \
  'habitat_baselines.eval.video_option=[disk]' \
  habitat.environment.max_episode_steps=${MAX_STEPS} \
  habitat.simulator.create_renderer=True \
  habitat.simulator.scene_dataset=data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json \
  habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
  habitat_baselines.torch_gpu_id=1 \
  habitat_baselines.rl.policy.name=OracleFBEPolicy \
  habitat_baselines.video_dir="${VIDEO_DIR}" \
  habitat_baselines.tensorboard_dir=tb/vlfm_split_gpu_video \
  > "${LOG}" 2>&1
EXIT=$?

echo "EXIT=${EXIT}  (124 = trainer-side timeout, 0 = clean exit)"
echo "-- tap counts --"
rg -o '\[PROBE [^]]*tag=([A-Z_]+)' -r '$1' "${LOG}" 2>/dev/null | sort | uniq -c | sort -nr | head -10 || true
echo "-- num distinct SIM (rgb,depth) tuples --"
rg -F 'tag=SIM ' "${LOG}" | rg -oP 'rgb_md5=\w+ \| depth_md5=\w+' | sort -u | wc -l || true
echo "-- episode done lines --"
grep -E "done=True|episode .* done|episode .* finish|num_episodes_finished" "${LOG}" | tail -5 || true
echo "-- last 5 AGENT_ACT --"
grep -F 'tag=AGENT_ACT' "${LOG}" | tail -5 || true
echo "-- video files written --"
ls -lh "${VIDEO_DIR}/" 2>/dev/null || true
echo "DONE"
