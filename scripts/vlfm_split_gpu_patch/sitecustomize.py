"""Split-GPU device-routing patch for VLFM's HabitatITMPolicyV2.

Auto-loaded by Python when this directory is on ``PYTHONPATH`` (Python's
built-in ``sitecustomize`` hook).  Pure runtime monkey-patch -- VLFM source
is not modified.

Why this exists
---------------
On a split-GPU layout (sim renderer on ``cuda:0``, torch actor on
``cuda:1`` via ``habitat_baselines.torch_gpu_id=1``), three hardcoded
``device="cuda"`` literals in VLFM's policy code resolve to ``cuda:0`` and
clash with the main actor's tensors on ``cuda:1``.  This raises
``Expected all tensors to be on the same device, but found at least two
devices, cuda:1 and cuda:0!`` the first time ``HabitatITMPolicyV2`` enters
its internal PointNav sub-policy.

Audited literals (read-only, NOT edited):

    vlfm/policy/utils/pointnav_policy.py:61  WrappedPointNavResNetPolicy.__init__ default device="cuda"
    vlfm/policy/base_objectnav_policy.py:255 _pointnav builds `masks`     with device="cuda"
    vlfm/policy/base_objectnav_policy.py:264 _pointnav builds `rho_theta` with device="cuda"

What we do
----------
When ``vlfm.policy.utils.pointnav_policy`` is imported, we:

1. Call ``torch.cuda.set_device(VLFM_POINTNAV_GPU_ID)`` in both the patch
   thread and the ``WrappedPointNavResNetPolicy.__init__`` thread so any
   later ``device="cuda"`` literal on those threads resolves to the same
   GPU as the main actor (``set_device`` is per-thread).
2. Wrap ``WrappedPointNavResNetPolicy.__init__`` to override its default
   ``device="cuda"`` argument with ``cuda:${VLFM_POINTNAV_GPU_ID}``.
3. Rebind ``pointnav_policy.move_obs_to_device`` so already-Tensor inputs
   are also ``.to(device)``-coerced (the upstream version only moves numpy
   arrays, leaving the ``rho_theta`` literal stranded on ``cuda:0``).
4. Wrap ``WrappedPointNavResNetPolicy.act`` to ``.to(self.device)`` the
   ``masks`` tensor that ``base_objectnav_policy._pointnav`` builds with
   a literal ``device="cuda"``.

Env vars:

    VLFM_POINTNAV_GPU_ID        target cuda index for the PointNav sub-policy.
                                Defaults to 1.  On a single-GPU box this is
                                effectively a no-op.

    VLFM_SKIP_SUCCESS_VIDEOS    1 => monkey-patch habitat_baselines'
                                generate_video to skip writing videos for
                                successful episodes (metrics["success"] >= 0.5).
                                Default 0 (write all).  Useful for full-val eval
                                where only failure videos are needed.

This is pure device routing + IO short-circuit -- no algorithm change.
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
import time


_TARGET_GPU_ID = int(os.environ.get("VLFM_POINTNAV_GPU_ID", "1"))
_INSTALLED = False
_LOCK = threading.Lock()


def _log(msg: str) -> None:
    sys.stderr.write(f"[VLFM_SPLIT_GPU_PATCH pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def _patch_pointnav_device() -> bool:
    """Install the four-layer device-routing patch.

    Returns True once the patch is installed (or determined to be a no-op
    because torch / the target GPU is unavailable); returns False if the
    target module is not yet imported and we should retry.
    """
    global _INSTALLED
    if _INSTALLED:
        return True

    mod = sys.modules.get("vlfm.policy.utils.pointnav_policy")
    if mod is None:
        return False
    cls = getattr(mod, "WrappedPointNavResNetPolicy", None)
    if cls is None:
        return False

    try:
        import torch
    except Exception as e:
        _log(f"torch import failed, patch is a no-op: {e!r}")
        _INSTALLED = True
        return True

    if not torch.cuda.is_available() or torch.cuda.device_count() <= _TARGET_GPU_ID:
        _log(
            f"cuda unavailable or device_count<={_TARGET_GPU_ID}, "
            f"patch is a no-op"
        )
        _INSTALLED = True
        return True

    with contextlib.suppress(Exception):
        prev = torch.cuda.current_device()
        torch.cuda.set_device(_TARGET_GPU_ID)
        _log(
            f"set_device: prev_current={prev} new_current={_TARGET_GPU_ID} "
            f"device_count={torch.cuda.device_count()}"
        )

    orig_init = cls.__init__
    if not getattr(orig_init, "_split_gpu_patched", False):

        def patched_init(self, ckpt_path, device="cuda", *a, **kw):
            if isinstance(device, str) and device == "cuda":
                device = f"cuda:{_TARGET_GPU_ID}"
            with contextlib.suppress(Exception):
                torch.cuda.set_device(_TARGET_GPU_ID)
            return orig_init(self, ckpt_path, device=device, *a, **kw)

        patched_init._split_gpu_patched = True  # type: ignore[attr-defined]
        cls.__init__ = patched_init
        _log("installed WrappedPointNavResNetPolicy.__init__")

    orig_move = getattr(mod, "move_obs_to_device", None)
    if orig_move is not None and not getattr(orig_move, "_split_gpu_patched", False):
        import numpy as _np

        def patched_move(observations, device, unsqueeze=False):
            for k, v in list(observations.items()):
                if isinstance(v, _np.ndarray):
                    tdtype = torch.uint8 if v.dtype == _np.uint8 else torch.float32
                    t = torch.from_numpy(v).to(device=device, dtype=tdtype)
                elif isinstance(v, torch.Tensor):
                    t = v if v.device == torch.device(device) else v.to(device=device)
                else:
                    continue
                if unsqueeze:
                    t = t.unsqueeze(0)
                observations[k] = t
            return observations

        patched_move._split_gpu_patched = True  # type: ignore[attr-defined]
        mod.move_obs_to_device = patched_move
        _log("installed pointnav_policy.move_obs_to_device")

    orig_act = cls.act
    if not getattr(orig_act, "_split_gpu_patched", False):

        def patched_act(self, observations, masks, deterministic=False):
            if isinstance(masks, torch.Tensor) and masks.device != self.device:
                masks = masks.to(self.device)
            return orig_act(self, observations, masks, deterministic=deterministic)

        patched_act._split_gpu_patched = True  # type: ignore[attr-defined]
        cls.act = patched_act
        _log("installed WrappedPointNavResNetPolicy.act")

    _INSTALLED = True
    return True


# --- ListConfig readonly fix for habitat-baselines' construct_envs ---------
# habitat_baselines.common.construct_vector_env.construct_envs does
#   scenes = config.habitat.dataset.content_scenes
#   ...
#   random.shuffle(scenes)
# which fails with `ReadonlyConfigError: ListConfig is read-only` when the
# scenes list comes from a Hydra `habitat.dataset.content_scenes=[a,b]`
# override (Hydra wraps overrides in a readonly ListConfig).  The default
# "*" path is fine because it goes through dataset.get_scenes_to_load(),
# which returns a fresh Python list.
#
# This patch wraps construct_envs to convert content_scenes to a plain
# Python list before calling the original.  No algorithm change; identical
# to the upstream behaviour once shuffle becomes legal.

_CONSTRUCT_ENVS_PATCHED = False


def _patch_construct_envs_listconfig() -> bool:
    """Replace habitat_baselines' construct_envs with a copy whose only
    change is `scenes = list(...)` so random.shuffle() doesn't trip
    OmegaConf's readonly guard on Hydra-overridden content_scenes.

    The original function is structurally simple; this patch reproduces it
    verbatim with the one-line fix.  Any habitat-baselines version drift
    on this function would silently bypass the upstream behaviour change,
    so we log the original module path for auditability.
    """
    global _CONSTRUCT_ENVS_PATCHED
    if _CONSTRUCT_ENVS_PATCHED:
        return True
    mod = sys.modules.get("habitat_baselines.common.construct_vector_env")
    if mod is None:
        return False
    orig = getattr(mod, "construct_envs", None)
    if orig is None:
        return False
    if getattr(orig, "_listconfig_patched", False):
        _CONSTRUCT_ENVS_PATCHED = True
        return True

    try:
        import random as _random
        from typing import Any as _Any, List as _List, Type as _Type
        from habitat import (
            ThreadedVectorEnv as _ThreadedVectorEnv,
            VectorEnv as _VectorEnv,
            logger as _logger,
            make_dataset as _make_dataset,
        )
        from habitat.config import read_write as _read_write
        from habitat.gym import make_gym_from_config as _make_gym_from_config
    except Exception as e:
        _log(f"construct_envs patch import failed: {e!r}")
        return True  # mark done, fall back to upstream

    def patched_construct_envs(
        config,
        workers_ignore_signals: bool = False,
        enforce_scenes_greater_eq_environments: bool = False,
    ):
        num_environments = config.habitat_baselines.num_environments
        configs = []
        dataset = _make_dataset(config.habitat.dataset.type)
        # Single-line departure from upstream: force a plain Python list so the
        # subsequent random.shuffle() doesn't fault on ReadonlyConfigError
        # when content_scenes came from a Hydra override.
        scenes = list(config.habitat.dataset.content_scenes)
        if "*" in scenes:
            scenes = list(dataset.get_scenes_to_load(config.habitat.dataset))

        if num_environments < 1:
            raise RuntimeError("num_environments must be strictly positive")

        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being "
                "able to split scenes uniquely between processes"
            )

        _random.shuffle(scenes)

        scene_splits = [[] for _ in range(num_environments)]  # type: _List[_List[str]]
        if len(scenes) < num_environments:
            msg = (
                f"There are less scenes ({len(scenes)}) than environments "
                f"({num_environments}). "
            )
            if enforce_scenes_greater_eq_environments:
                _logger.warn(
                    msg
                    + "Reducing the number of environments to be the number of scenes."
                )
                num_environments = len(scenes)
                scene_splits = [[s] for s in scenes]
            else:
                _logger.warn(
                    msg
                    + "Each environment will use all the scenes instead of using a subset."
                )
            for scene in scenes:
                for split in scene_splits:
                    split.append(scene)
        else:
            for idx, scene in enumerate(scenes):
                scene_splits[idx % len(scene_splits)].append(scene)
            assert sum(map(len, scene_splits)) == len(scenes)

        for i in range(num_environments):
            proc_config = config.copy()
            with _read_write(proc_config):
                task_config = proc_config.habitat
                task_config.seed = task_config.seed + i
                if len(scenes) > 0:
                    task_config.dataset.content_scenes = scene_splits[i]
            configs.append(proc_config)

        vector_env_cls: _Type[_Any]
        if int(os.environ.get("HABITAT_ENV_DEBUG", 0)):
            _logger.warn(
                "Using the debug Vector environment interface. Expect slower performance."
            )
            vector_env_cls = _ThreadedVectorEnv
        else:
            vector_env_cls = _VectorEnv

        return vector_env_cls(
            make_env_fn=_make_gym_from_config,
            env_fn_args=tuple((c,) for c in configs),
            workers_ignore_signals=workers_ignore_signals,
        )

    patched_construct_envs._listconfig_patched = True  # type: ignore[attr-defined]
    mod.construct_envs = patched_construct_envs
    # ppo_trainer did `from ... import construct_envs`, rebind its name too.
    ppo = sys.modules.get("habitat_baselines.rl.ppo.ppo_trainer")
    if ppo is not None and hasattr(ppo, "construct_envs"):
        ppo.construct_envs = patched_construct_envs
    _CONSTRUCT_ENVS_PATCHED = True
    _log("installed construct_envs ListConfig readonly fix (full rewrite)")
    return True


# --- Fail-only video patch -------------------------------------------------
# When VLFM_SKIP_SUCCESS_VIDEOS=1, short-circuit
# habitat_baselines.utils.common.generate_video for episodes where
# metrics["success"] >= 0.5 so only failure mp4s are written to disk.
# Filename-based selection isn't enough -- the trainer writes every episode
# unconditionally and that's wasted IO / disk on long eval runs.
#
# Env var:
#     VLFM_SKIP_SUCCESS_VIDEOS    1 => skip success videos, 0/unset => write all.

_SKIP_SUCCESS_VIDEOS = bool(int(os.environ.get("VLFM_SKIP_SUCCESS_VIDEOS", "0")))
_VIDEO_PATCH_INSTALLED = False


def _patch_video_skip() -> bool:
    """Wrap habitat_baselines.utils.common.generate_video so success-case
    episodes are not flushed to disk.  Returns True when installed (or no-op);
    False if the target module isn't loaded yet and we should retry."""
    global _VIDEO_PATCH_INSTALLED
    if _VIDEO_PATCH_INSTALLED:
        return True
    if not _SKIP_SUCCESS_VIDEOS:
        _VIDEO_PATCH_INSTALLED = True
        return True

    mod = sys.modules.get("habitat_baselines.utils.common")
    if mod is None:
        return False
    orig = getattr(mod, "generate_video", None)
    if orig is None:
        return False
    if getattr(orig, "_fail_only_patched", False):
        _VIDEO_PATCH_INSTALLED = True
        return True

    def patched_generate_video(*args, **kwargs):
        metrics = kwargs.get("metrics", None)
        if metrics is None and len(args) >= 6:
            metrics = args[5]
        if isinstance(metrics, dict):
            success = metrics.get("success", 0.0)
            try:
                if float(success) >= 0.5:
                    return
            except (TypeError, ValueError):
                pass
        return orig(*args, **kwargs)

    patched_generate_video._fail_only_patched = True  # type: ignore[attr-defined]
    mod.generate_video = patched_generate_video
    # ppo_trainer did `from ... import generate_video`, so the name binding
    # inside that module also has to be rewritten or the trainer keeps calling
    # the original.
    ppo = sys.modules.get("habitat_baselines.rl.ppo.ppo_trainer")
    if ppo is not None and hasattr(ppo, "generate_video"):
        ppo.generate_video = patched_generate_video
    _VIDEO_PATCH_INSTALLED = True
    _log("installed generate_video fail-only patch (VLFM_SKIP_SUCCESS_VIDEOS=1)")
    return True


def _patch_loop() -> None:
    """Poll for the three target modules to be imported and install the
    patches as soon as they appear.  Times out after 180 s."""
    deadline = time.time() + 180.0
    pointnav_done = False
    video_done = False
    scenes_done = False
    while time.time() < deadline:
        with _LOCK:
            if not pointnav_done:
                pointnav_done = _patch_pointnav_device()
            if not video_done:
                video_done = _patch_video_skip()
            if not scenes_done:
                scenes_done = _patch_construct_envs_listconfig()
        if pointnav_done and video_done and scenes_done:
            return
        time.sleep(0.2)
    if not pointnav_done:
        _log("install timeout: vlfm.policy.utils.pointnav_policy never imported")
    if not video_done:
        _log("install timeout: habitat_baselines.utils.common never imported")
    if not scenes_done:
        _log("install timeout: habitat_baselines.common.construct_vector_env never imported")


def _after_fork_in_child() -> None:
    global _INSTALLED, _VIDEO_PATCH_INSTALLED, _CONSTRUCT_ENVS_PATCHED
    _INSTALLED = False
    _VIDEO_PATCH_INSTALLED = False
    _CONSTRUCT_ENVS_PATCHED = False
    _log(f"fork: ppid={os.getppid()}, restarting patch loop")
    threading.Thread(target=_patch_loop, daemon=True).start()


threading.Thread(target=_patch_loop, daemon=True).start()

with contextlib.suppress(Exception):
    os.register_at_fork(after_in_child=_after_fork_in_child)

_log(
    f"boot: VLFM_POINTNAV_GPU_ID={_TARGET_GPU_ID} "
    f"VLFM_SKIP_SUCCESS_VIDEOS={int(_SKIP_SUCCESS_VIDEOS)} "
    f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}"
)
