#!/usr/bin/env bash
# Trainer launch with HARD GPU isolation between habitat-sim (EGL/GL render)
# and torch (policy.act CUDA kernels), to prove out the fix for the renderer
# freeze diagnosed in outputs/var_J_torch_min_*.log + outputs/var_H_*_*.log.
#
# Why this should fix it:
#   - On the SAME physical GPU, ANY torch CUDA kernel issued between two
#     habitat_sim.Simulator.step() calls deadlocks the EGL/GL renderer on
#     nvidia driver 570 + EGL surfaceless + H20-3e
#     (var_J_torch_min_same_gpu.log: depth_uniq=1/5, avg 15.7s/step,
#     classic 20s EGL fence watchdog recovery).
#   - On DIFFERENT physical GPUs the freeze vanishes
#     (var_J_torch_min_split_gpu.log: depth_uniq=5/5, avg 0.005s/step;
#      var_H_torch_interleave_split_gpu.log: 5x 512x512 matmul also PASS).
#   - The trainer process puts sim and policy in the same process by design
#     (ThreadedVectorEnv when HABITAT_ENV_DEBUG=1; default forkserver
#      VectorEnv has its own EGL-init worker hang on this box, see
#      outputs/vlfm_no_envdebug.log).  We KEEP HABITAT_ENV_DEBUG=1 so we
#      stay single-process and known-debuggable, then move sim and torch
#      to different physical GPUs so they no longer fight.
#
# Mapping:
#   CUDA_VISIBLE_DEVICES=4,5  -> cuda:0 == physical 4, cuda:1 == physical 5
#   habitat.simulator.habitat_sim_v0.gpu_device_id=0  -> sim renderer on GPU 4
#   habitat_baselines.torch_gpu_id=1                  -> policy.act on GPU 5
#
# Success criteria (read from outputs/vlfm_split_gpu.log after the run):
#   - tag=SIM tap shows >= 6 distinct (rgb_md5, depth_md5) tuples (no freeze)
#   - SIM_CFG shows gpu_device_id=0 (i.e. CUDA_VISIBLE_DEVICES mapped GPU 4)
#   - Episode actually completes (success / metric line near the end), or
#     at minimum N tap-SIM lines == N tap-POLICY lines and the trainer keeps
#     making forward progress until the 360s timeout.
set -u

cd "$(dirname "$0")/.."
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_cuda_sim

# Two clean GPUs (see nvidia-smi pre-flight in run_split_gpu_probe.sh output:
# both GPU 4 and GPU 5 have ~143 GiB free).
export CUDA_VISIBLE_DEVICES=4,5

export PYTHONPATH=outputs/probe_pkg
export HF_ENDPOINT=https://hf-mirror.com

# Stay on the single-process / ThreadedVectorEnv path on purpose.  We've
# proved cross-GPU isolation makes this path safe; the default forkserver
# path has a separate EGL-init hang on this box that is NOT what we are
# fixing here.
export HABITAT_ENV_DEBUG=1

python -c "
import habitat_sim, torch
print('[precheck] habitat_sim', habitat_sim.__version__,
      'cuda=', habitat_sim.cuda_enabled,
      'bullet=', habitat_sim.built_with_bullet,
      'path=', habitat_sim.__file__)
print('[precheck] torch', torch.__version__,
      'cuda.device_count=', torch.cuda.device_count(),
      'visible=', __import__('os').environ.get('CUDA_VISIBLE_DEVICES'))
assert habitat_sim.cuda_enabled, 'wrong env (need habitat_sim with cuda_enabled)'
assert torch.cuda.device_count() >= 2, 'need >=2 visible GPUs for split-gpu isolation'
"

LOG=outputs/vlfm_split_gpu.log
TIMEOUT_S=360

echo "================================================================"
echo "vlfm.run, split-GPU isolation (sim=cuda:0, torch=cuda:1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (physical 4 + 5)"
echo "HABITAT_ENV_DEBUG=${HABITAT_ENV_DEBUG} (ThreadedVectorEnv)"
echo "timeout=${TIMEOUT_S}s   log=${LOG}"
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
  habitat_baselines.video_dir=video_dir/vlfm_split_gpu \
  habitat_baselines.tensorboard_dir=tb/vlfm_split_gpu \
  > "${LOG}" 2>&1
EXIT=$?

echo "EXIT=${EXIT}  (124 = trainer-side timeout, 0 = clean exit)"
echo "-- tap counts --"
rg -o '\[PROBE [^]]*tag=([A-Z_]+)' -r '$1' "${LOG}" 2>/dev/null | sort | uniq -c | sort -nr | head -10 || true
echo "-- SIM_CFG (relevant flags) --"
rg -F 'tag=SIM_CFG' "${LOG}" | head -1 || true
echo "-- distinct (rgb_md5, depth_md5) seen at SIM tap (should be many) --"
rg -F 'tag=SIM ' "${LOG}" | rg -oP 'rgb_md5=\w+ \| depth_md5=\w+' | sort -u | head -20 || true
echo "-- num distinct SIM (rgb,depth) tuples --"
rg -F 'tag=SIM ' "${LOG}" | rg -oP 'rgb_md5=\w+ \| depth_md5=\w+' | sort -u | wc -l || true
echo "-- last 12 SIM lines --"
rg -F 'tag=SIM ' "${LOG}" | rg -oP 'step=\d+|rgb_md5=\w+|depth_md5=\w+|agent_rot=q\(w=[-0-9.]+' | paste -d' ' - - - - - | tail -12 || true
echo "-- Traceback / AssertionError --"
rg -F 'Traceback' "${LOG}" | head -3 || true
rg -F 'AssertionError' "${LOG}" | head -3 || true
echo "-- episode completion / metrics --"
rg -iE 'episode .* done|episode_reward|spl|success|metric' "${LOG}" | tail -5 || true
echo "DONE"
