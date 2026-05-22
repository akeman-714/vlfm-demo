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

Env var:

    VLFM_POINTNAV_GPU_ID    target cuda index for the PointNav sub-policy.
                            Defaults to 1.  On a single-GPU box this is
                            effectively a no-op.

This is pure device routing -- no algorithm change.
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


def _patch_loop() -> None:
    """Poll for ``vlfm.policy.utils.pointnav_policy`` to be imported and
    install the patch as soon as it appears.  Times out after 180 s."""
    deadline = time.time() + 180.0
    while time.time() < deadline:
        with _LOCK:
            if _patch_pointnav_device():
                return
        time.sleep(0.2)
    _log("install timeout: vlfm.policy.utils.pointnav_policy never imported")


def _after_fork_in_child() -> None:
    global _INSTALLED
    _INSTALLED = False
    _log(f"fork: ppid={os.getppid()}, restarting patch loop")
    threading.Thread(target=_patch_loop, daemon=True).start()


threading.Thread(target=_patch_loop, daemon=True).start()

with contextlib.suppress(Exception):
    os.register_at_fork(after_in_child=_after_fork_in_child)

_log(
    f"boot: VLFM_POINTNAV_GPU_ID={_TARGET_GPU_ID} "
    f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}"
)
