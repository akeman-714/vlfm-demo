"""Auto-loaded by Python at startup when this directory is on PYTHONPATH.

Probe taps along the entire observation pipeline:

  [SIM]              habitat_sim.Simulator.get_sensor_observations  (subprocess)
  [TASK]             habitat.core.embodied_task.EmbodiedTask.step    (subprocess)
  [ENV_STEP]         habitat.core.env.Env.step                       (subprocess)
  [WRAP_STEP]        habitat.gym.gym_env_obs_dict_wrapper.EnvObsDictWrapper.step  (subprocess)
  [PIPE_WRITE]       VectorEnv._worker_env's connection_write_fn     (subprocess)
  [VENV_STEP]        VectorEnv.step                                  (main)
  [BATCH_IN/OUT]     habitat_baselines.utils.common.batch_obs        (main)
  [TRANSFORMS]       apply_obs_transforms_batch                      (main)
  [POLICY]           HabitatMixin._cache_observations                (main)

We DO NOT change any execution logic.  All taps print structured
[PROBE pid=...|tag=...|step=...|...]  lines.

We try to attach patches early via a background thread that polls for the
relevant modules to be imported.  This is safe for the existing flow
because we only monkey-patch a few methods to add logging around the
original behavior.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import sys
import threading
import time
import traceback

import numpy as np


# ---- per-process state -------------------------------------------------
_STEP = {
    "sim": 0,           # subprocess sim sensor reads
    "task": 0,          # subprocess task.step
    "env": 0,           # subprocess Env.step
    "wrap": 0,          # subprocess EnvObsDictWrapper.step
    "venv": 0,          # main VectorEnv.step
    "batch": 0,         # main batch_obs
    "tx": 0,            # main apply_obs_transforms_batch
    "policy": 0,        # main HabitatMixin._cache_observations
}
_TAPS_INSTALLED = set()
_LOCK = threading.Lock()


def _md5(arr) -> str:
    try:
        if hasattr(arr, "cpu"):
            arr = arr.detach().cpu().numpy()
        b = np.ascontiguousarray(arr).tobytes()
        return hashlib.md5(b).hexdigest()[:10]
    except Exception as e:
        return f"<err:{e!r}>"


def _stat(arr):
    try:
        if hasattr(arr, "cpu"):
            arr = arr.detach().cpu().numpy()
        a = np.asarray(arr)
        return f"shape={tuple(a.shape)} dtype={a.dtype} mean={float(a.mean()):.4f} min={float(a.min()):.4f} max={float(a.max()):.4f}"
    except Exception as e:
        return f"<stat err:{e!r}>"


def _print(msg: str) -> None:
    # one print per line, immediate flush, atomic enough for our purposes
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _log(tag: str, **fields) -> None:
    parts = [f"pid={os.getpid()}", f"tag={tag}"]
    for k, v in fields.items():
        parts.append(f"{k}={v}")
    _print("[PROBE " + " | ".join(parts) + "]")


# ---- helpers to extract rgb/depth from various containers --------------
def _extract_rgb_depth_from_obs(obs):
    """Return (rgb_arr, depth_arr) or (None, None) if not present.

    Handles:
      - dict with "rgb"/"depth"
      - dict wrapped under "obs"
      - per-agent dict: {agent_id_int: {"rgb":..., "depth":...}}
    """
    if obs is None:
        return None, None
    if isinstance(obs, dict):
        if "rgb" in obs or "depth" in obs:
            return obs.get("rgb"), obs.get("depth")
        # per-agent: pick agent 0
        if 0 in obs and isinstance(obs[0], dict):
            sub = obs[0]
            return sub.get("rgb"), sub.get("depth")
        if "obs" in obs and isinstance(obs["obs"], dict):
            return obs["obs"].get("rgb"), obs["obs"].get("depth")
        # try the first value if it's a dict
        try:
            first = next(iter(obs.values()))
            if isinstance(first, dict):
                return first.get("rgb"), first.get("depth")
        except Exception:
            pass
        return None, None
    if hasattr(obs, "__getitem__"):
        with contextlib.suppress(Exception):
            return obs["rgb"], obs["depth"]
    return None, None


def _extract_rgb_depth_from_tensordict(td, idx=0):
    """For main-process batched TensorDict obs."""
    try:
        rgb = td["rgb"][idx]
    except Exception:
        rgb = None
    try:
        depth = td["depth"][idx]
    except Exception:
        depth = None
    return rgb, depth


# ---- patches -----------------------------------------------------------
def _patch_agent_act():
    """Hook habitat_sim.Agent.act to confirm rotations are dispatched."""
    if "agent_act" in _TAPS_INSTALLED:
        return False
    mod = sys.modules.get("habitat_sim.agent.agent") or sys.modules.get("habitat_sim")
    Agent = None
    if mod is not None:
        Agent = getattr(mod, "Agent", None)
    if Agent is None:
        # try the agent.agent submodule
        try:
            from habitat_sim.agent import agent as _aa  # type: ignore
            Agent = getattr(_aa, "Agent", None)
        except Exception:
            pass
    if Agent is None:
        return False
    orig = Agent.act
    if getattr(orig, "_probe_patched", False):
        _TAPS_INSTALLED.add("agent_act")
        return True

    def patched(self, action_id):
        pre_pos = pre_rot = "?"
        try:
            st = self.get_state()
            pre_pos = f"[{st.position[0]:.3f},{st.position[1]:.3f},{st.position[2]:.3f}]"
            pre_rot = f"q(w={st.rotation.w:.4f},y={st.rotation.y:.4f})"
        except Exception:
            pass
        out = orig(self, action_id)
        try:
            st = self.get_state()
            _log(
                "AGENT_ACT",
                action_id=str(action_id),
                pre_pos=pre_pos,
                pre_rot=pre_rot,
                post_pos=f"[{st.position[0]:.3f},{st.position[1]:.3f},{st.position[2]:.3f}]",
                post_rot=f"q(w={st.rotation.w:.4f},y={st.rotation.y:.4f})",
                collided=str(out),
            )
        except Exception as e:
            _log("AGENT_ACT_ERR", err=repr(e))
        return out

    patched._probe_patched = True  # type: ignore[attr-defined]
    Agent.act = patched
    _TAPS_INSTALLED.add("agent_act")
    _log("INSTALL", which="Agent.act")
    return True


def _patch_sensor_draw_and_read():
    """Disabled — wrapping Sensor.draw_observation / get_observation appears to
    interfere with habitat-sim's renderer timing.  Marker keeps the patch loop
    from blocking on this tap forever.
    """
    _TAPS_INSTALLED.add("sensor_io")
    return True


def _patch_sim_get_sensor_observations():
    if "sim" in _TAPS_INSTALLED:
        return False
    mod = sys.modules.get("habitat_sim")
    if mod is None:
        return False
    Sim = getattr(mod, "Simulator", None)
    if Sim is None or not hasattr(Sim, "get_sensor_observations"):
        return False
    # Patch __init__ to log thread that creates the simulator (GL context will
    # belong to this thread).
    orig_init = Sim.__init__
    if not getattr(orig_init, "_probe_patched", False):
        def patched_init(self, *args, **kwargs):
            _log(
                "SIM_INIT_PRE",
                tid=threading.get_ident(),
                tname=threading.current_thread().name,
            )
            out = orig_init(self, *args, **kwargs)
            _log(
                "SIM_INIT_POST",
                tid=threading.get_ident(),
                tname=threading.current_thread().name,
            )
            return out

        patched_init._probe_patched = True  # type: ignore[attr-defined]
        Sim.__init__ = patched_init

    orig = Sim.get_sensor_observations
    if getattr(orig, "_probe_patched", False):
        _TAPS_INSTALLED.add("sim")
        return True

    def patched(self, *args, **kwargs):
        # tid lets us see if get_sensor_observations is being called from a
        # different thread than the one that created the GL context (a known
        # source of "frozen render" bugs with thread-local EGL contexts).
        _tid = threading.get_ident()
        _tname = threading.current_thread().name
        # Once, dump resolved sim_config so we can see what create_renderer,
        # requires_textures, gpu_gpu, scene_id, etc. ended up as.
        if _STEP["sim"] == 0:
            try:
                cfg = self.config
                sim_cfg = cfg.sim_cfg
                fields = []
                for k in (
                    "scene_id",
                    "scene_dataset_config_file",
                    "gpu_device_id",
                    "create_renderer",
                    "requires_textures",
                    "load_semantic_mesh",
                    "enable_physics",
                    "enable_gfx_replay_save",
                    "allow_sliding",
                    "frustum_culling",
                    "leave_context_with_background_renderer",
                    "enable_batch_renderer",
                ):
                    v = getattr(sim_cfg, k, "?")
                    fields.append(f"{k}={v}")
                for ai, ac in enumerate(cfg.agents):
                    fields.append(f"agent{ai}_n_sensors={len(ac.sensor_specifications)}")
                    for ss in ac.sensor_specifications:
                        try:
                            fields.append(
                                f"sensor.{ss.uuid}.type={ss.sensor_type}/res={list(ss.resolution)}/pos={list(ss.position)}/hfov={getattr(ss,'hfov','?')}/gpu2gpu={getattr(ss,'gpu2gpu_transfer','?')}"
                            )
                        except Exception as e:
                            fields.append(f"sensor.???={e!r}")
                _log("SIM_CFG", **{f"f{i}": v for i, v in enumerate(fields)})
            except Exception as e:
                _log("SIM_CFG_ERR", err=repr(e))

        # capture pre-call agent + sensor states
        pre_sensor = "?"
        try:
            ag0 = self.get_agent(0)
            st0 = ag0.get_state()
            try:
                ss = st0.sensor_states  # dict[str, SensorState]
                items = []
                for name, sst in ss.items():
                    p = sst.position
                    r = sst.rotation
                    items.append(
                        f"{name}@(p=[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}] q(w={r.w:.3f},y={r.y:.3f}))"
                    )
                pre_sensor = ",".join(items)
            except Exception:
                pre_sensor = "<no sensor_states>"
        except Exception:
            pass
        out = orig(self, *args, **kwargs)
        try:
            n = _STEP["sim"]
            agent = self.get_agent(0)
            st = agent.get_state()
            pos = st.position
            rot = st.rotation
            pos_str = f"[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}]"
            rot_str = (
                f"q(w={rot.w:.4f},y={rot.y:.4f})"
            )
            rgb, depth = _extract_rgb_depth_from_obs(out)
            _log(
                "SIM",
                step=n,
                tid=_tid,
                tname=_tname,
                rgb_md5=_md5(rgb) if rgb is not None else "none",
                depth_md5=_md5(depth) if depth is not None else "none",
                depth_stat=_stat(depth) if depth is not None else "none",
                agent_pos=pos_str,
                agent_rot=rot_str,
                pre_sensor=pre_sensor,
            )
            _STEP["sim"] += 1
        except Exception as e:
            _log("SIM_ERR", err=repr(e))
        return out

    patched._probe_patched = True  # type: ignore[attr-defined]
    Sim.get_sensor_observations = patched
    _TAPS_INSTALLED.add("sim")
    _log("INSTALL", which="sim.get_sensor_observations")
    return True


def _patch_task_step():
    if "task" in _TAPS_INSTALLED:
        return False
    mod = sys.modules.get("habitat.core.embodied_task")
    if mod is None:
        return False
    Task = getattr(mod, "EmbodiedTask", None)
    if Task is None:
        return False
    orig = Task.step
    if getattr(orig, "_probe_patched", False):
        _TAPS_INSTALLED.add("task")
        return True

    def patched(self, action, episode):
        out = orig(self, action, episode)
        try:
            n = _STEP["task"]
            rgb, depth = _extract_rgb_depth_from_obs(out)
            _log(
                "TASK",
                step=n,
                action=str(action),
                rgb_md5=_md5(rgb) if rgb is not None else "none",
                depth_md5=_md5(depth) if depth is not None else "none",
                keys=str(list(out.keys()) if hasattr(out, "keys") else []),
            )
            _STEP["task"] += 1
        except Exception as e:
            _log("TASK_ERR", err=repr(e))
        return out

    patched._probe_patched = True  # type: ignore[attr-defined]
    Task.step = patched
    _TAPS_INSTALLED.add("task")
    _log("INSTALL", which="EmbodiedTask.step")
    return True


def _patch_env_step():
    if "env" in _TAPS_INSTALLED:
        return False
    mod = sys.modules.get("habitat.core.env")
    if mod is None:
        return False
    Env = getattr(mod, "Env", None)
    if Env is None:
        return False
    orig = Env.step
    if getattr(orig, "_probe_patched", False):
        _TAPS_INSTALLED.add("env")
        return True

    def patched(self, action, **kwargs):
        out = orig(self, action, **kwargs)
        try:
            n = _STEP["env"]
            rgb, depth = _extract_rgb_depth_from_obs(out)
            _log(
                "ENV",
                step=n,
                action=str(action),
                rgb_md5=_md5(rgb) if rgb is not None else "none",
                depth_md5=_md5(depth) if depth is not None else "none",
            )
            _STEP["env"] += 1
        except Exception as e:
            _log("ENV_ERR", err=repr(e))
        return out

    patched._probe_patched = True  # type: ignore[attr-defined]
    Env.step = patched
    _TAPS_INSTALLED.add("env")
    _log("INSTALL", which="Env.step")
    return True


def _patch_wrap_step():
    if "wrap" in _TAPS_INSTALLED:
        return False
    mod = sys.modules.get("habitat.gym.gym_env_obs_dict_wrapper")
    if mod is None:
        return False
    W = getattr(mod, "EnvObsDictWrapper", None)
    if W is None:
        return False
    orig = W.step
    if getattr(orig, "_probe_patched", False):
        _TAPS_INSTALLED.add("wrap")
        return True

    def patched(self, action):
        out = orig(self, action)
        try:
            n = _STEP["wrap"]
            obs, reward, done, info = out
            rgb, depth = _extract_rgb_depth_from_obs(obs)
            _log(
                "WRAP",
                step=n,
                action=str(action),
                rgb_md5=_md5(rgb) if rgb is not None else "none",
                depth_md5=_md5(depth) if depth is not None else "none",
                done=str(done),
                rgb_id=id(rgb) if rgb is not None else "none",
                depth_id=id(depth) if depth is not None else "none",
            )
            _STEP["wrap"] += 1
        except Exception as e:
            _log("WRAP_ERR", err=repr(e))
        return out

    patched._probe_patched = True  # type: ignore[attr-defined]
    W.step = patched
    _TAPS_INSTALLED.add("wrap")
    _log("INSTALL", which="EnvObsDictWrapper.step")
    return True


def _patch_venv_step():
    if "venv" in _TAPS_INSTALLED:
        return False
    mod = sys.modules.get("habitat.core.vector_env")
    if mod is None:
        return False
    VE = getattr(mod, "VectorEnv", None)
    if VE is None:
        return False
    orig_step = VE.step
    if getattr(orig_step, "_probe_patched", False):
        _TAPS_INSTALLED.add("venv")
        return True

    def patched_step(self, data):
        try:
            _log("VENV_PRE", step=_STEP["venv"], n_envs=self.num_envs, data=str(data))
        except Exception:
            pass
        out = orig_step(self, data)
        try:
            n = _STEP["venv"]
            for i, (obs, r, d, info) in enumerate(out):
                rgb, depth = _extract_rgb_depth_from_obs(obs)
                _log(
                    "VENV",
                    step=n,
                    env=i,
                    rgb_md5=_md5(rgb) if rgb is not None else "none",
                    depth_md5=_md5(depth) if depth is not None else "none",
                    done=str(d),
                    rgb_id=id(rgb) if rgb is not None else "none",
                )
            _STEP["venv"] += 1
        except Exception as e:
            _log("VENV_ERR", err=repr(e))
        return out

    patched_step._probe_patched = True  # type: ignore[attr-defined]
    VE.step = patched_step
    # Also patch ThreadedVectorEnv (subclass override of step)
    TVE = getattr(mod, "ThreadedVectorEnv", None)
    if TVE is not None and TVE.step is not patched_step:
        # If ThreadedVectorEnv has its own step(), patch it too
        try:
            orig_tstep = TVE.step
            if not getattr(orig_tstep, "_probe_patched", False):
                def patched_tstep(self, data):
                    out = orig_tstep(self, data)
                    try:
                        n = _STEP["venv"]
                        for i, (obs, r, d, info) in enumerate(out):
                            rgb, depth = _extract_rgb_depth_from_obs(obs)
                            _log(
                                "TVENV",
                                step=n,
                                env=i,
                                rgb_md5=_md5(rgb) if rgb is not None else "none",
                                depth_md5=_md5(depth) if depth is not None else "none",
                                done=str(d),
                            )
                        _STEP["venv"] += 1
                    except Exception as e:
                        _log("TVENV_ERR", err=repr(e))
                    return out
                patched_tstep._probe_patched = True  # type: ignore[attr-defined]
                TVE.step = patched_tstep
        except Exception as e:
            _log("TVENV_PATCH_ERR", err=repr(e))
    _TAPS_INSTALLED.add("venv")
    _log("INSTALL", which="VectorEnv.step")
    return True


def _patch_batch_obs():
    if "batch" in _TAPS_INSTALLED:
        return False
    mod = sys.modules.get("habitat_baselines.utils.common")
    if mod is None:
        return False
    orig = getattr(mod, "batch_obs", None)
    if orig is None or getattr(orig, "_probe_patched", False):
        if orig is not None:
            _TAPS_INSTALLED.add("batch")
        return False

    def patched(observations, device=None):
        try:
            n = _STEP["batch"]
            for i, o in enumerate(observations):
                rgb, depth = _extract_rgb_depth_from_obs(o)
                _log(
                    "BATCH_IN",
                    step=n,
                    env=i,
                    rgb_md5=_md5(rgb) if rgb is not None else "none",
                    depth_md5=_md5(depth) if depth is not None else "none",
                )
        except Exception as e:
            _log("BATCH_IN_ERR", err=repr(e))
        out = orig(observations, device=device)
        try:
            n = _STEP["batch"]
            rgb_t, depth_t = _extract_rgb_depth_from_tensordict(out, 0)
            _log(
                "BATCH_OUT",
                step=n,
                rgb_md5=_md5(rgb_t) if rgb_t is not None else "none",
                depth_md5=_md5(depth_t) if depth_t is not None else "none",
                rgb_dataptr=hex(rgb_t.data_ptr()) if rgb_t is not None and hasattr(rgb_t, "data_ptr") else "n/a",
                depth_dataptr=hex(depth_t.data_ptr()) if depth_t is not None and hasattr(depth_t, "data_ptr") else "n/a",
            )
            _STEP["batch"] += 1
        except Exception as e:
            _log("BATCH_OUT_ERR", err=repr(e))
        return out

    patched._probe_patched = True  # type: ignore[attr-defined]
    mod.batch_obs = patched
    # also try to rebind in ppo_trainer module's local imports
    pt = sys.modules.get("habitat_baselines.rl.ppo.ppo_trainer")
    if pt is not None:
        try:
            pt.batch_obs = patched
        except Exception:
            pass
    _TAPS_INSTALLED.add("batch")
    _log("INSTALL", which="batch_obs")
    return True


def _patch_transforms():
    if "tx" in _TAPS_INSTALLED:
        return False
    mod = sys.modules.get(
        "habitat_baselines.common.obs_transformers"
    )
    if mod is None:
        return False
    orig = getattr(mod, "apply_obs_transforms_batch", None)
    if orig is None:
        return False
    if getattr(orig, "_probe_patched", False):
        _TAPS_INSTALLED.add("tx")
        return True

    def patched(batch, transforms):
        try:
            n = _STEP["tx"]
            rgb_t, depth_t = _extract_rgb_depth_from_tensordict(batch, 0)
            _log(
                "TX_IN",
                step=n,
                rgb_md5=_md5(rgb_t) if rgb_t is not None else "none",
                depth_md5=_md5(depth_t) if depth_t is not None else "none",
                n_tx=len(transforms),
            )
        except Exception as e:
            _log("TX_IN_ERR", err=repr(e))
        out = orig(batch, transforms)
        try:
            n = _STEP["tx"]
            rgb_t, depth_t = _extract_rgb_depth_from_tensordict(out, 0)
            _log(
                "TX_OUT",
                step=n,
                rgb_md5=_md5(rgb_t) if rgb_t is not None else "none",
                depth_md5=_md5(depth_t) if depth_t is not None else "none",
            )
            _STEP["tx"] += 1
        except Exception as e:
            _log("TX_OUT_ERR", err=repr(e))
        return out

    patched._probe_patched = True  # type: ignore[attr-defined]
    mod.apply_obs_transforms_batch = patched
    pt = sys.modules.get("habitat_baselines.rl.ppo.ppo_trainer")
    if pt is not None:
        try:
            pt.apply_obs_transforms_batch = patched
        except Exception:
            pass
    _TAPS_INSTALLED.add("tx")
    _log("INSTALL", which="apply_obs_transforms_batch")
    return True


def _patch_policy_cache():
    if "policy" in _TAPS_INSTALLED:
        return False
    mod = sys.modules.get("vlfm.policy.habitat_policies")
    if mod is None:
        return False
    cls = getattr(mod, "HabitatMixin", None)
    if cls is None:
        return False
    orig = cls._cache_observations
    if getattr(orig, "_probe_patched", False):
        _TAPS_INSTALLED.add("policy")
        return True

    def patched(self, observations):
        try:
            n = _STEP["policy"]
            rgb_t, depth_t = _extract_rgb_depth_from_tensordict(observations, 0)
            cache_len = len(self._observations_cache)
            _log(
                "POLICY",
                step=n,
                cache_len=cache_len,
                rgb_md5=_md5(rgb_t) if rgb_t is not None else "none",
                depth_md5=_md5(depth_t) if depth_t is not None else "none",
                depth_stat=_stat(depth_t) if depth_t is not None else "none",
            )
            _STEP["policy"] += 1
        except Exception as e:
            _log("POLICY_ERR", err=repr(e))
        return orig(self, observations)

    patched._probe_patched = True  # type: ignore[attr-defined]
    cls._cache_observations = patched
    _TAPS_INSTALLED.add("policy")
    _log("INSTALL", which="HabitatMixin._cache_observations")
    return True


def _patch_pointnav_device():
    """Force the internal PointNav sub-policy onto the same CUDA device as the
    main torch actor.

    Why this exists:
      VLFM has several hardcoded `device="cuda"` literals that torch
      resolves to cuda:0.  On a split-GPU layout (sim renderer on
      cuda:0 / phys 4, torch actor on cuda:1 / phys 5 via
      `habitat_baselines.torch_gpu_id=1`), the main actor tensors live
      on cuda:1 but these literal-"cuda" tensors end up on cuda:0,
      triggering:
        "Expected all tensors to be on the same device, but found at
         least two devices, cuda:1 and cuda:0!"
      during the first explore-phase pointnav call.

      Affected spots (read-only audit, NOT edited):
        - WrappedPointNavResNetPolicy.__init__ default device="cuda"
        - base_objectnav_policy._pointnav line 255  (masks tensor)
        - base_objectnav_policy._pointnav line 264  (rho_theta tensor)

    What this does:
      1. Calls `torch.cuda.set_device(VLFM_POINTNAV_GPU_ID)` so that any
         later `device="cuda"` literal resolves to the same physical GPU
         as the main actor.  This fixes the two _pointnav literals
         without touching vlfm/policy/.
      2. Wraps `WrappedPointNavResNetPolicy.__init__` to pass a fully
         specified `cuda:${VLFM_POINTNAV_GPU_ID}` device, belt-and-
         suspenders in case set_device wasn't honored (e.g. trainer
         flips it later).

    Controlled by env var VLFM_POINTNAV_GPU_ID (default 1).  No change
    to policy algorithm; this is pure device routing.
    """
    if "pointnav_device" in _TAPS_INSTALLED:
        return False
    mod = sys.modules.get("vlfm.policy.utils.pointnav_policy")
    if mod is None:
        return False
    cls = getattr(mod, "WrappedPointNavResNetPolicy", None)
    if cls is None:
        return False
    orig_init = cls.__init__
    if getattr(orig_init, "_probe_patched", False):
        _TAPS_INSTALLED.add("pointnav_device")
        return True

    target_gpu_id = int(os.environ.get("VLFM_POINTNAV_GPU_ID", "1"))

    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > target_gpu_id:
            prev = torch.cuda.current_device()
            torch.cuda.set_device(target_gpu_id)
            _log(
                "POINTNAV_SET_DEVICE",
                prev_current=prev,
                new_current=target_gpu_id,
                device_count=torch.cuda.device_count(),
            )
    except Exception as e:
        _log("POINTNAV_SET_DEVICE_ERR", err=repr(e))

    def patched_init(self, ckpt_path, device="cuda", *a, **kw):
        if isinstance(device, str) and device == "cuda":
            device = f"cuda:{target_gpu_id}"
            _log("POINTNAV_DEVICE_OVERRIDE", forced_to=device, ckpt=ckpt_path)
        # set_device is per-thread; also force it inside the calling
        # thread so any later `device="cuda"` literal in policy code that
        # runs on the SAME thread resolves to the target GPU.
        with contextlib.suppress(Exception):
            torch.cuda.set_device(target_gpu_id)
        return orig_init(self, ckpt_path, device=device, *a, **kw)

    patched_init._probe_patched = True  # type: ignore[attr-defined]
    cls.__init__ = patched_init
    _TAPS_INSTALLED.add("pointnav_device")
    _log("INSTALL", which="WrappedPointNavResNetPolicy.__init__")

    # ---- also rebind move_obs_to_device to coerce ALL tensor inputs
    # (not just numpy arrays) onto self.device.  This handles the
    # `torch.tensor(..., device="cuda")` literals in
    # base_objectnav_policy._pointnav (rho_theta_tensor) that produce
    # cuda:0 tensors even when current_device is 1, because they ran on
    # a different thread.
    orig_move = getattr(mod, "move_obs_to_device", None)
    if orig_move is not None and not getattr(orig_move, "_probe_patched", False):
        import numpy as _np

        def patched_move(observations, device, unsqueeze=False):
            try:
                import torch as _torch

                for k, v in list(observations.items()):
                    if isinstance(v, _np.ndarray):
                        tdtype = _torch.uint8 if v.dtype == _np.uint8 else _torch.float32
                        t = _torch.from_numpy(v).to(device=device, dtype=tdtype)
                    elif isinstance(v, _torch.Tensor):
                        if v.device != _torch.device(device):
                            t = v.to(device=device)
                        else:
                            t = v
                    else:
                        continue
                    if unsqueeze:
                        t = t.unsqueeze(0)
                    observations[k] = t
            except Exception as e:
                _log("MOVE_OBS_ERR", err=repr(e))
            return observations

        patched_move._probe_patched = True  # type: ignore[attr-defined]
        mod.move_obs_to_device = patched_move
        _log("INSTALL", which="pointnav_policy.move_obs_to_device")

    # ---- wrap WrappedPointNavResNetPolicy.act so the `masks` Tensor is
    # also forced onto self.device.  Same root cause: _pointnav creates
    # masks with hardcoded device="cuda".
    orig_act = cls.act
    if not getattr(orig_act, "_probe_patched", False):

        def patched_act(self, observations, masks, deterministic=False):
            try:
                import torch as _torch

                if isinstance(masks, _torch.Tensor) and masks.device != self.device:
                    masks = masks.to(self.device)
            except Exception as e:
                _log("MASKS_MOVE_ERR", err=repr(e))
            return orig_act(self, observations, masks, deterministic=deterministic)

        patched_act._probe_patched = True  # type: ignore[attr-defined]
        cls.act = patched_act
        _log("INSTALL", which="WrappedPointNavResNetPolicy.act")
    return True


# ---- patch loop --------------------------------------------------------
def _try_patches() -> None:
    # All taps are independent — install whichever modules are present.
    with contextlib.suppress(Exception):
        _patch_sim_get_sensor_observations()
    with contextlib.suppress(Exception):
        _patch_agent_act()
    with contextlib.suppress(Exception):
        _patch_sensor_draw_and_read()
    with contextlib.suppress(Exception):
        _patch_task_step()
    with contextlib.suppress(Exception):
        _patch_env_step()
    with contextlib.suppress(Exception):
        _patch_wrap_step()
    with contextlib.suppress(Exception):
        _patch_venv_step()
    with contextlib.suppress(Exception):
        _patch_batch_obs()
    with contextlib.suppress(Exception):
        _patch_transforms()
    with contextlib.suppress(Exception):
        _patch_policy_cache()
    with contextlib.suppress(Exception):
        _patch_pointnav_device()


def _patch_loop() -> None:
    deadline = time.time() + 180.0
    last_count = -1
    while time.time() < deadline:
        with _LOCK:
            _try_patches()
            count = len(_TAPS_INSTALLED)
        if count != last_count:
            _log("INSTALL_STATUS", taps=str(sorted(_TAPS_INSTALLED)))
            last_count = count
        if count >= 11:
            return
        time.sleep(0.2)
    _log("INSTALL_TIMEOUT", taps=str(sorted(_TAPS_INSTALLED)))


def _start_thread():
    threading.Thread(target=_patch_loop, daemon=True).start()


def _after_fork_in_child():
    # The polling thread does not survive fork/forkserver.  Restart it
    # and clear "already-installed" markers so we re-attach to the
    # patched-by-parent-but-not-by-us modules if they were rebound.
    _STEP.update({k: 0 for k in _STEP})
    _TAPS_INSTALLED.clear()
    _log("FORK_CHILD", ppid=os.getppid())
    _start_thread()


_start_thread()

with contextlib.suppress(Exception):
    os.register_at_fork(after_in_child=_after_fork_in_child)

_log(
    "BOOT",
    PYTHONPATH=os.environ.get("PYTHONPATH", ""),
    cmdline=" ".join(sys.argv),
    HABITAT_ENV_DEBUG=os.environ.get("HABITAT_ENV_DEBUG", "<unset>"),
)
