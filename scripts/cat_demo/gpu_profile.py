#!/usr/bin/env python3
"""Per-PID x per-GPU VRAM sampler for the cat-demo edge-deployment study.

Goal
----
Answer "if this stack were on a real robot (no Habitat), how much VRAM does it
need?" by running ONE real ``find_cat`` episode and attributing every MiB to a
role.  The demo already splits GPUs (Habitat/EGL on the sim card, PointNav on
the nav card, the 4 VLM servers on their own card[s]), so a single instrumented
run yields BOTH the with-Habitat number and the real-env inference-only number
-- no separate no-Habitat run required.

Two independent attribution methods are recorded so the report can cross-check:

  * per-PID  (``nvidia-smi --query-compute-apps``)  -- the only correct way on
    the *shared* VLM card(s) (GPU0/GPU7), where other tenants co-exist.
  * per-card (``nvidia-smi --query-gpu``)           -- clean on the *empty* sim
    and nav cards, where (card_total - baseline) == that role's footprint even
    if habitat-sim's EGL context does not surface as a CUDA compute-app.

Roles
-----
  yolov7 / mobilesam / blip2itm / groundingdino   VLM servers (cmdline match)
  pointnav                                        vlfm.run on the nav card
  habitat_sim                                     vlfm.run on the sim card
  other                                           any other CUDA proc (logged, excluded)

Outputs (under --out)
---------------------
  meta.json          run metadata (GPU model/driver, card split, command, phases)
  timeline_proc.csv  per (sample, pid, gpu) rows: role + used_mib
  timeline_gpu.csv   per (sample, gpu) rows: mem_used/free + util

Depth->pointcloud, obstacle/value/frontier maps are numpy/CPU (0 torch refs);
they cost ~0 VRAM and are asserted (not sampled) by the report.

Usage
-----
  python scripts/cat_demo/gpu_profile.py --sim-card 1 --nav-card 5 \
      --out outputs/gpu_profile_$(date +%Y%m%d_%H%M%S)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_SCRIPT = os.path.join(REPO_DIR, "scripts", "cat_demo", "eval_cat_demo.sh")

# cmdline substring -> role.  Order matters (first hit wins).
_ROLE_PATTERNS: List[Tuple[str, str]] = [
    ("vlfm.vlm.grounding_dino", "groundingdino"),
    ("vlfm.vlm.blip2itm", "blip2itm"),
    ("vlfm.vlm.yolov7", "yolov7"),
    ("vlfm.vlm.sam", "mobilesam"),
    # vlfm.run is split into pointnav/habitat_sim by GPU index at sample time.
    ("vlfm.run", "vlfm_run"),
]

_cmdline_cache: Dict[int, str] = {}


def _smi(query: str, extra: str = "") -> List[str]:
    """Run nvidia-smi --query-<...> --format=csv,noheader,nounits -> lines."""
    cmd = ["nvidia-smi", query, "--format=csv,noheader,nounits"]
    if extra:
        cmd += extra.split()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def gpu_uuid_index_map() -> Dict[str, int]:
    m: Dict[str, int] = {}
    for ln in _smi("--query-gpu=index,uuid"):
        idx, uuid = [x.strip() for x in ln.split(",")][:2]
        m[uuid] = int(idx)
    return m


def gpu_static_info() -> List[Dict[str, str]]:
    info = []
    for ln in _smi("--query-gpu=index,name,memory.total,driver_version"):
        idx, name, mem_total, driver = [x.strip() for x in ln.split(",")][:4]
        info.append({"index": idx, "name": name, "memory_total_mib": mem_total, "driver": driver})
    return info


def pid_cmdline(pid: int) -> str:
    if pid in _cmdline_cache:
        return _cmdline_cache[pid]
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except Exception:
        raw = ""
    if raw:  # only cache once we have a readable cmdline
        _cmdline_cache[pid] = raw
    return raw


def role_for(pid: int, gpu_index: int, sim_idx: int, nav_idx: int) -> str:
    cmd = pid_cmdline(pid)
    if not cmd:
        return "other"
    for needle, role in _ROLE_PATTERNS:
        if needle in cmd:
            if role == "vlfm_run":
                if gpu_index == sim_idx:
                    return "habitat_sim"
                if gpu_index == nav_idx:
                    return "pointnav"
                return "vlfm_run_other"
            return role
    return "other"


def sample_procs(uuid2idx: Dict[str, int], sim_idx: int, nav_idx: int) -> List[Tuple[int, int, str, int]]:
    """-> list of (gpu_index, pid, role, used_mib)."""
    rows = []
    for ln in _smi("--query-compute-apps=gpu_uuid,pid,used_memory"):
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid_s, mem_s = parts[0], parts[1], parts[2]
        if uuid not in uuid2idx:
            continue
        try:
            pid, mem = int(pid_s), int(float(mem_s))
        except ValueError:
            continue
        gidx = uuid2idx[uuid]
        rows.append((gidx, pid, role_for(pid, gidx, sim_idx, nav_idx), mem))
    return rows


def sample_gpu() -> List[Tuple[int, int, int, int]]:
    """-> list of (gpu_index, mem_used_mib, mem_free_mib, util_pct)."""
    rows = []
    for ln in _smi("--query-gpu=index,memory.used,memory.free,utilization.gpu"):
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < 4:
            continue
        try:
            rows.append((int(parts[0]), int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))))
        except ValueError:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-card", type=int, required=True, help="physical GPU index for Habitat/EGL (cuda:0)")
    ap.add_argument("--nav-card", type=int, required=True, help="physical GPU index for PointNav/torch (cuda:1)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--interval", type=float, default=0.2, help="sample period seconds (default 0.2 = 5Hz)")
    ap.add_argument("--baseline-sec", type=float, default=6.0, help="idle sampling before launching the episode")
    ap.add_argument("--cooldown-sec", type=float, default=4.0, help="sampling after the episode exits")
    ap.add_argument("--max-sec", type=float, default=1200.0, help="hard cap on episode duration")
    ap.add_argument("--split", default="cat_demo", help="habitat eval split (default cat_demo = find_cat)")
    ap.add_argument("--n-ep", type=int, default=1, help="episode count (default 1 -> exactly one nav)")
    ap.add_argument("--no-run", action="store_true", help="only snapshot current state; do not launch an episode")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found", file=sys.stderr)
        return 2

    uuid2idx = gpu_uuid_index_map()
    sim_idx, nav_idx = args.sim_card, args.nav_card

    proc_csv = open(os.path.join(out, "timeline_proc.csv"), "w", buffering=1)
    gpu_csv = open(os.path.join(out, "timeline_gpu.csv"), "w", buffering=1)
    proc_csv.write("ts,elapsed_s,phase,gpu_index,pid,role,used_mib\n")
    gpu_csv.write("ts,elapsed_s,phase,gpu_index,mem_used_mib,mem_free_mib,util_pct\n")

    t0 = time.time()

    def snap(phase: str) -> None:
        ts = time.time()
        el = ts - t0
        for gidx, pid, role, mem in sample_procs(uuid2idx, sim_idx, nav_idx):
            proc_csv.write(f"{ts:.3f},{el:.3f},{phase},{gidx},{pid},{role},{mem}\n")
        for gidx, used, free, util in sample_gpu():
            gpu_csv.write(f"{ts:.3f},{el:.3f},{phase},{gidx},{used},{free},{util}\n")

    def sample_for(seconds: float, phase: str, stop_proc: Optional[subprocess.Popen] = None) -> None:
        end = time.time() + seconds
        while time.time() < end:
            snap(phase)
            if stop_proc is not None and stop_proc.poll() is not None:
                return
            time.sleep(args.interval)

    # ---- metadata ----
    meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "repo_dir": REPO_DIR,
        "sim_card": sim_idx,
        "nav_card": nav_idx,
        "interval_s": args.interval,
        "split": args.split,
        "n_ep": args.n_ep,
        "gpus": gpu_static_info(),
        "roles": {
            "inference": ["yolov7", "mobilesam", "blip2itm", "groundingdino", "pointnav"],
            "sim": ["habitat_sim"],
            "cpu_zero_vram": ["depth_to_pointcloud", "obstacle_map", "value_map", "frontier_map"],
        },
        "note": "depth->pointcloud and the maps are numpy/CPU (0 torch refs) -> ~0 VRAM.",
    }

    print(f">>> gpu_profile  sim=GPU{sim_idx}  nav=GPU{nav_idx}  out={out}")
    print(f">>> baseline {args.baseline_sec}s ...")
    sample_for(args.baseline_sec, "baseline")

    episode_log = os.path.join(out, "episode.log")
    if not args.no_run:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = f"{sim_idx},{nav_idx}"
        env["VLFM_POINTNAV_GPU_ID"] = "1"  # cuda:1 == nav card
        env["SPLIT"] = args.split
        env["N_EP"] = str(args.n_ep)
        env["VIDEO_DIR"] = os.path.join(out, "video")
        env["TB_DIR"] = os.path.join(out, "tb")
        env["LOG"] = episode_log
        meta["episode_cmd"] = f"CUDA_VISIBLE_DEVICES={sim_idx},{nav_idx} VLFM_POINTNAV_GPU_ID=1 bash {EVAL_SCRIPT}"
        meta["episode_log"] = episode_log
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f">>> launching episode (log -> {episode_log}) ...")
        proc = subprocess.Popen(["bash", EVAL_SCRIPT], cwd=REPO_DIR, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ep_start = time.time()
        try:
            while proc.poll() is None:
                snap("episode")
                if time.time() - ep_start > args.max_sec:
                    print(f">>> max-sec {args.max_sec}s hit, terminating episode", file=sys.stderr)
                    proc.send_signal(signal.SIGINT)
                    time.sleep(5)
                    if proc.poll() is None:
                        proc.kill()
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
        meta["episode_exit"] = proc.returncode
        meta["episode_seconds"] = round(time.time() - ep_start, 1)
        print(f">>> episode exit={proc.returncode} dur={meta['episode_seconds']}s")
    else:
        meta["episode_cmd"] = "(--no-run: snapshot only)"

    print(f">>> cooldown {args.cooldown_sec}s ...")
    sample_for(args.cooldown_sec, "cooldown")

    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    meta["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    proc_csv.close()
    gpu_csv.close()
    print(f">>> done. wrote timeline_proc.csv, timeline_gpu.csv, meta.json under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
