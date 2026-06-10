#!/usr/bin/env python3
"""Analyse a gpu_profile.py run and emit per-component VRAM tables + charts.

Reads ``timeline_proc.csv`` / ``timeline_gpu.csv`` / ``meta.json`` from a
gpu_profile output dir and produces:

  summary.json / summary.md   per-component idle/peak/steady + two roll-ups
                              (with / without GroundingDINO) + Habitat separated
  chart1_peak_bars.png        per-component peak VRAM (inference vs sim coloured)
  chart2_stacked_area.png     VRAM over the episode, inference stack + Habitat
  chart3_rollups.png          总计(含GDINO) / 总计(不含GDINO) / Habitat / 完整demo
  chart4_gpu_util.png         per-card GPU utilisation over the episode

Component accounting
--------------------
  VLM servers (yolov7/mobilesam/blip2itm/groundingdino) -> per-PID (shared card).
  pointnav / habitat_sim                                -> per-card delta on the
        empty nav/sim cards (card_used - baseline), which is robust even if the
        EGL renderer never surfaces as a CUDA compute-app.  Per-PID is kept as a
        cross-check.
  depth->pointcloud + maps                              -> 0 VRAM (numpy/CPU),
        asserted from source (0 torch refs), shown as a labelled 0 line.

Usage:  python scripts/cat_demo/gpu_report.py --in outputs/gpu_profile_<ts>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MIB_PER_GB = 1024.0

# ---- CJK font (fall back to English labels if unavailable) ----------------
_FONT_GLOBS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc",
    "/usr/share/fonts/**/NotoSansCJK*.ttc",
    "/usr/share/fonts/**/NotoSansCJK*.otf",
]


def _setup_font() -> bool:
    from matplotlib import font_manager as fm

    for pat in _FONT_GLOBS:
        for path in sorted(glob.glob(pat, recursive=True)):
            try:
                fm.fontManager.addfont(path)
                name = fm.FontProperties(fname=path).get_name()
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return True
            except Exception:
                continue
    plt.rcParams["axes.unicode_minus"] = False
    return False


USE_CJK = _setup_font()


def L(zh: str, en: str) -> str:
    return zh if USE_CJK else en


# role -> (zh, en, is_inference)
ROLE_META = {
    "yolov7": ("YOLOv7", "YOLOv7", True),
    "groundingdino": ("GroundingDINO", "GroundingDINO", True),
    "mobilesam": ("MobileSAM", "MobileSAM", True),
    "blip2itm": ("BLIP2-ITM", "BLIP2-ITM", True),
    "pointnav": ("导航 PointNav", "nav (PointNav)", True),
    "depth_to_pointcloud": ("深度→点云(CPU)", "depth→pointcloud (CPU)", True),
    "habitat_sim": ("Habitat 仿真", "Habitat sim", False),
}
VLM_ROLES = ["yolov7", "groundingdino", "mobilesam", "blip2itm"]


def disp(role: str) -> str:
    zh, en, _ = ROLE_META[role]
    return L(zh, en)


def _med(s: pd.Series) -> float:
    return float(s.median()) if len(s) else 0.0


def _max(s: pd.Series) -> float:
    return float(s.max()) if len(s) else 0.0


def analyse(indir: str) -> Dict:
    proc = pd.read_csv(os.path.join(indir, "timeline_proc.csv"))
    gpu = pd.read_csv(os.path.join(indir, "timeline_gpu.csv"))
    with open(os.path.join(indir, "meta.json")) as f:
        meta = json.load(f)
    sim_idx, nav_idx = meta["sim_card"], meta["nav_card"]

    base_p = proc[proc.phase == "baseline"]
    epi_p = proc[proc.phase == "episode"]
    base_g = gpu[gpu.phase == "baseline"]
    epi_g = gpu[gpu.phase == "episode"]

    comp: Dict[str, Dict[str, float]] = {}

    # VLM servers: per-PID
    for role in VLM_ROLES:
        comp[role] = {
            "idle_mib": _med(base_p[base_p.role == role].used_mib),
            "peak_mib": _max(epi_p[epi_p.role == role].used_mib),
            "steady_mib": _med(epi_p[epi_p.role == role].used_mib),
            "method": "per-pid",
        }

    # pointnav / habitat: per-card delta on the (empty) nav/sim cards
    def card_delta(card: int) -> Dict[str, float]:
        b = _med(base_g[base_g.gpu_index == card].mem_used_mib)
        es = epi_g[epi_g.gpu_index == card].mem_used_mib
        return {
            "idle_mib": 0.0,
            "peak_mib": max(0.0, _max(es) - b),
            "steady_mib": max(0.0, _med(es) - b),
            "method": "per-card-delta",
            "baseline_card_mib": b,
        }

    comp["pointnav"] = card_delta(nav_idx)
    comp["habitat_sim"] = card_delta(sim_idx)
    # per-PID cross-check for the vlfm.run split
    comp["pointnav"]["peak_pid_mib"] = _max(epi_p[epi_p.role == "pointnav"].used_mib)
    comp["habitat_sim"]["peak_pid_mib"] = _max(epi_p[epi_p.role == "habitat_sim"].used_mib)

    # depth->pointcloud: 0 VRAM (CPU/numpy)
    comp["depth_to_pointcloud"] = {"idle_mib": 0.0, "peak_mib": 0.0, "steady_mib": 0.0, "method": "cpu-asserted"}

    def peak(role: str) -> float:
        return comp[role]["peak_mib"]

    infer_with = VLM_ROLES + ["pointnav"]  # depth = 0
    total_with = sum(peak(r) for r in infer_with)
    total_without = total_with - peak("groundingdino")
    habitat = peak("habitat_sim")

    rollups = {
        "total_with_gdino_mib": total_with,
        "total_without_gdino_mib": total_without,
        "habitat_sim_mib": habitat,
        "full_demo_with_habitat_mib": total_with + habitat,
    }

    # observed concurrent inference peak (sum across VLM roles at same ts) + nav
    concurrent_vlm = 0.0
    if len(epi_p):
        piv = epi_p[epi_p.role.isin(VLM_ROLES)].groupby("ts").used_mib.sum()
        concurrent_vlm = float(piv.max()) if len(piv) else 0.0
    rollups["observed_concurrent_vlm_mib"] = concurrent_vlm
    rollups["observed_concurrent_with_nav_mib"] = concurrent_vlm + peak("pointnav")

    return {"meta": meta, "components": comp, "rollups": rollups,
            "frames": {"proc": proc, "gpu": gpu}, "cards": (sim_idx, nav_idx)}


# ---------------------------------------------------------------- charts ----
def _gb(mib: float) -> str:
    return f"{mib:.0f} MiB ({mib / MIB_PER_GB:.2f} GB)"


def chart_peak_bars(res: Dict, path: str) -> None:
    comp = res["components"]
    order = ["depth_to_pointcloud", "mobilesam", "groundingdino", "yolov7", "pointnav", "blip2itm", "habitat_sim"]
    names = [disp(r) for r in order]
    vals = [comp[r]["peak_mib"] for r in order]
    colors = ["#9ecae1" if ROLE_META[r][2] else "#fdae6b" for r in order]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(order))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(L("峰值显存 (MiB)", "peak VRAM (MiB)"))
    ax.set_title(L("各组件峰值显存（蓝=推理栈，橙=仿真，端侧可减）",
                   "Per-component peak VRAM (blue=inference, orange=sim, droppable)"))
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, f" {v:.0f} ({v / MIB_PER_GB:.2f}G)", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18 if max(vals) else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_stacked_area(res: Dict, path: str) -> None:
    proc, gpu = res["frames"]["proc"], res["frames"]["gpu"]
    sim_idx, nav_idx = res["cards"]
    epi_g = gpu[gpu.phase == "episode"]
    if not len(epi_g):
        return
    grid = np.sort(epi_g[epi_g.gpu_index == sim_idx].elapsed_s.values)
    if len(grid) < 2:
        grid = np.sort(epi_g.elapsed_s.unique())
    epi_p = proc[proc.phase == "episode"]

    def series_proc(role: str) -> np.ndarray:
        r = epi_p[epi_p.role == role].sort_values("ts")
        if not len(r):
            return np.zeros_like(grid)
        return np.interp(grid, r.elapsed_s.values, r.used_mib.values)

    def series_card_delta(card: int) -> np.ndarray:
        base = res["components"]["pointnav" if card == nav_idx else "habitat_sim"].get("baseline_card_mib", 0.0)
        g = epi_g[epi_g.gpu_index == card].sort_values("elapsed_s")
        if not len(g):
            return np.zeros_like(grid)
        return np.clip(np.interp(grid, g.elapsed_s.values, g.mem_used_mib.values) - base, 0, None)

    stack_roles = ["blip2itm", "yolov7", "groundingdino", "mobilesam"]
    stacks = [series_proc(r) for r in stack_roles]
    stacks.append(series_card_delta(nav_idx))  # pointnav
    labels = [disp(r) for r in stack_roles] + [disp("pointnav")]
    colors = ["#3182bd", "#6baed6", "#9ecae1", "#c6dbef", "#74c476"]

    habitat = series_card_delta(sim_idx)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.stackplot(grid, *stacks, labels=labels, colors=colors)
    infer_total = np.sum(stacks, axis=0)
    ax.plot(grid, infer_total + habitat, color="#e6550d", lw=2.0, ls="--",
            label=L("+Habitat 仿真", "+Habitat sim"))
    ax.set_xlabel(L("时间 (秒)", "elapsed (s)"))
    ax.set_ylabel(L("显存 (MiB)", "VRAM (MiB)"))
    ax.set_title(L("一次 find_cat episode 的显存占用（堆叠=推理栈；虚线含仿真）",
                   "VRAM over one find_cat episode (stack=inference; dashed=+sim)"))
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.set_ylim(0, (np.max(infer_total + habitat) if len(grid) else 1) * 1.15 + 1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_rollups(res: Dict, path: str) -> None:
    r = res["rollups"]
    keys = ["total_without_gdino_mib", "total_with_gdino_mib", "habitat_sim_mib", "full_demo_with_habitat_mib"]
    names = [
        L("总计\n(不含GDINO)", "total\n(no GDINO)"),
        L("总计\n(含GDINO)", "total\n(+GDINO)"),
        L("Habitat仿真\n(可减掉)", "Habitat\n(droppable)"),
        L("完整demo\n(含Habitat)", "full demo\n(+Habitat)"),
    ]
    vals = [r[k] for k in keys]
    colors = ["#31a354", "#74c476", "#fdae6b", "#969696"]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(keys))
    ax.bar(x, vals, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(L("峰值显存 (MiB)", "peak VRAM (MiB)"))
    ax.set_title(L("端侧部署显存口径对比", "Edge-deployment VRAM roll-ups"))
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.01, f"{v:.0f}\n{v / MIB_PER_GB:.2f} GB", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.16 if max(vals) else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_gpu_util(res: Dict, path: str) -> None:
    gpu = res["frames"]["gpu"]
    sim_idx, nav_idx = res["cards"]
    epi = gpu[gpu.phase == "episode"]
    if not len(epi):
        return
    fig, ax = plt.subplots(figsize=(10, 4.6))
    plotted = []
    for card, lbl, col in [
        (sim_idx, L(f"GPU{sim_idx} 仿真卡", f"GPU{sim_idx} sim"), "#e6550d"),
        (nav_idx, L(f"GPU{nav_idx} 导航卡", f"GPU{nav_idx} nav"), "#74c476"),
        (0, L("GPU0 VLM卡", "GPU0 VLM"), "#3182bd"),
        (7, L("GPU7 BLIP2卡", "GPU7 BLIP2"), "#756bb1"),
    ]:
        g = epi[epi.gpu_index == card].sort_values("elapsed_s")
        if len(g):
            ax.plot(g.elapsed_s.values, g.util_pct.values, label=lbl, color=col, lw=1.3)
            plotted.append(card)
    ax.set_xlabel(L("时间 (秒)", "elapsed (s)"))
    ax.set_ylabel(L("GPU 利用率 (%)", "GPU util (%)"))
    ax.set_ylim(-2, 102)
    ax.set_title(L("各卡 GPU 利用率（仿真卡 vs 推理卡占空比）",
                   "Per-card GPU utilisation (sim vs compute duty cycle)"))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_summary(res: Dict, indir: str) -> None:
    comp, roll, meta = res["components"], res["rollups"], res["meta"]
    out = {"meta": {k: meta.get(k) for k in
                    ("started", "sim_card", "nav_card", "split", "n_ep", "episode_exit", "episode_seconds")},
           "components_mib": comp, "rollups_mib": roll,
           "rollups_gb": {k: round(v / MIB_PER_GB, 2) for k, v in roll.items()}}
    with open(os.path.join(indir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    order = ["depth_to_pointcloud", "yolov7", "mobilesam", "blip2itm", "groundingdino", "pointnav", "habitat_sim"]
    lines = ["# Cat-demo GPU footprint (per component)", ""]
    g = res["meta"].get("gpus", [{}])[0]
    lines += [f"- GPU: {g.get('name','?')}  driver {g.get('driver','?')}  "
              f"total {g.get('memory_total_mib','?')} MiB",
              f"- sim card GPU{meta['sim_card']}, nav card GPU{meta['nav_card']}, "
              f"split={meta.get('split')}, n_ep={meta.get('n_ep')}, "
              f"episode {meta.get('episode_seconds','?')}s exit={meta.get('episode_exit','?')}",
              ""]
    lines += ["| component | idle MiB | peak MiB | peak GB | method |",
              "|---|---:|---:|---:|---|"]
    for r in order:
        c = comp[r]
        lines.append(f"| {disp(r)} | {c['idle_mib']:.0f} | {c['peak_mib']:.0f} | "
                     f"{c['peak_mib']/MIB_PER_GB:.2f} | {c['method']} |")
    lines += ["", "## Roll-ups", "",
              "| scope | MiB | GB |", "|---|---:|---:|",
              f"| total without GroundingDINO | {roll['total_without_gdino_mib']:.0f} | "
              f"{roll['total_without_gdino_mib']/MIB_PER_GB:.2f} |",
              f"| total with GroundingDINO | {roll['total_with_gdino_mib']:.0f} | "
              f"{roll['total_with_gdino_mib']/MIB_PER_GB:.2f} |",
              f"| Habitat sim (droppable) | {roll['habitat_sim_mib']:.0f} | "
              f"{roll['habitat_sim_mib']/MIB_PER_GB:.2f} |",
              f"| full demo (+Habitat) | {roll['full_demo_with_habitat_mib']:.0f} | "
              f"{roll['full_demo_with_habitat_mib']/MIB_PER_GB:.2f} |",
              "",
              "_depth→pointcloud + obstacle/value/frontier maps are numpy/CPU (0 torch refs) → ~0 VRAM._",
              "_VLM peaks are per-PID resident (each incl. its own ~0.5–0.8 GB CUDA context); "
              "consolidating into one process on edge removes (N−1) contexts._"]
    with open(os.path.join(indir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    args = ap.parse_args()
    indir = os.path.abspath(args.indir)
    res = analyse(indir)
    write_summary(res, indir)
    chart_peak_bars(res, os.path.join(indir, "chart1_peak_bars.png"))
    chart_stacked_area(res, os.path.join(indir, "chart2_stacked_area.png"))
    chart_rollups(res, os.path.join(indir, "chart3_rollups.png"))
    chart_gpu_util(res, os.path.join(indir, "chart4_gpu_util.png"))
    r = res["rollups"]
    print(f"CJK font: {'on' if USE_CJK else 'OFF (English labels)'}")
    print(f"without GDINO : {r['total_without_gdino_mib']:.0f} MiB "
          f"({r['total_without_gdino_mib']/MIB_PER_GB:.2f} GB)")
    print(f"with GDINO    : {r['total_with_gdino_mib']:.0f} MiB "
          f"({r['total_with_gdino_mib']/MIB_PER_GB:.2f} GB)")
    print(f"Habitat sim   : {r['habitat_sim_mib']:.0f} MiB "
          f"({r['habitat_sim_mib']/MIB_PER_GB:.2f} GB)")
    print(f"wrote summary.md/json + 4 charts under {indir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
