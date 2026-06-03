#!/usr/bin/env python3
"""Build a static index.html that groups+sorts VLFM episode mp4s by failure mode.

Usage:
    python3 scripts/build_video_index.py <root_dir>

<root_dir> should contain procA_video/ and/or procB_video/ subdirs holding mp4
files whose names match:
    episode=<id>-ckpt=0-distance_to_goal=<f>-success=<f>-spl=<f>-soft_spl=<f>
    -distance_to_goal_reward=<f>-traveled_stairs=<f>-yaw=<f>
    -target_detected=<f>-stop_called=<f>-start_yaw=<f>.mp4

Writes <root_dir>/index.html.
"""
from __future__ import annotations

import html
import os
import re
import sys
from datetime import datetime
from pathlib import Path

FIELDS = (
    "episode",
    "distance_to_goal",
    "success",
    "spl",
    "soft_spl",
    "target_detected",
    "stop_called",
    "yaw",
    "start_yaw",
)
NUM = r"-?\d+(?:\.\d+)?"
PATTERNS = {f: re.compile(rf"{f}=({NUM})") for f in FIELDS}


def parse_name(name: str) -> dict | None:
    out: dict = {"name": name}
    for k, pat in PATTERNS.items():
        m = pat.search(name)
        if not m:
            return None
        out[k] = float(m.group(1))
    out["episode"] = int(out["episode"])
    return out


def bucket(rec: dict) -> str:
    d = rec["distance_to_goal"]
    detected = rec["target_detected"] >= 0.5
    stopped = rec["stop_called"] >= 0.5
    succ = rec["success"] >= 0.5
    if succ:
        return "success"
    if stopped and d < 3.0:
        return "near_miss_stop"
    if stopped and d >= 3.0:
        return "false_positive_stop"
    if not stopped and detected and d >= 3.0:
        return "detected_no_stop"
    if not stopped and not detected and d >= 15.0:
        return "wandered_off"
    if not stopped and not detected:
        return "timeout_no_target"
    return "other"


BUCKET_LABEL = {
    "success": "Success",
    "near_miss_stop": "Near miss (stopped <3m)",
    "false_positive_stop": "False positive stop (>=3m)",
    "detected_no_stop": "Detected but never stopped",
    "wandered_off": "Wandered off (>=15m)",
    "timeout_no_target": "Timeout, never detected",
    "other": "Other",
}
BUCKET_ORDER = list(BUCKET_LABEL.keys())


def collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or not sub.name.endswith("_video"):
            continue
        proc = sub.name.replace("_video", "")
        for f in sorted(sub.iterdir()):
            if f.suffix != ".mp4":
                continue
            rec = parse_name(f.name)
            if rec is None:
                continue
            rec["proc"] = proc
            rec["rel_path"] = f"{sub.name}/{f.name}"
            rec["size_mb"] = f.stat().st_size / 1024 / 1024
            rec["bucket"] = bucket(rec)
            rows.append(rec)
    return rows


CSS = """
:root { color-scheme: dark light; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 16px; background: #111; color: #ddd; }
h1 { margin: 0 0 4px; font-size: 18px; }
.summary { color: #888; font-size: 13px; margin-bottom: 14px; }
.controls { position: sticky; top: 0; background: #111; padding: 8px 0 12px;
            border-bottom: 1px solid #333; z-index: 10; display: flex;
            gap: 10px; flex-wrap: wrap; align-items: center; }
.controls button { background: #222; color: #ddd; border: 1px solid #444;
                   padding: 4px 10px; border-radius: 4px; cursor: pointer;
                   font-size: 12px; }
.controls button.active { background: #2a5; border-color: #4c8; color: #fff; }
.controls select { background: #222; color: #ddd; border: 1px solid #444;
                   padding: 4px; border-radius: 4px; font-size: 12px; }
h2.bucket { margin: 18px 0 6px; font-size: 15px; color: #fa6;
            border-left: 3px solid #fa6; padding-left: 8px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
        gap: 12px; }
.card { background: #1b1b1b; border: 1px solid #2a2a2a; border-radius: 6px;
        padding: 8px; }
.card video { width: 100%; height: auto; background: #000; border-radius: 4px;
              display: block; }
.meta { font-size: 12px; margin-top: 6px; line-height: 1.4; }
.meta .tag { display: inline-block; padding: 1px 6px; border-radius: 3px;
             margin-right: 4px; font-size: 11px; }
.tag.detected { background: #284; color: #fff; }
.tag.no-detected { background: #443; color: #aaa; }
.tag.stopped { background: #826; color: #fff; }
.tag.no-stopped { background: #443; color: #aaa; }
.tag.proc { background: #246; color: #cfd; }
.dist { color: #fa6; font-weight: 600; }
.spl { color: #6cf; }
.empty { color: #666; font-size: 13px; padding: 8px 0; }
"""

JS = """
function setSort(by) {
  document.querySelectorAll('.grid').forEach(g => {
    const cards = Array.from(g.children);
    cards.sort((a,b) => {
      const av = parseFloat(a.dataset[by]);
      const bv = parseFloat(b.dataset[by]);
      if (isNaN(av)) return 1; if (isNaN(bv)) return -1;
      return by === 'episode' ? av - bv : av - bv;
    });
    cards.forEach(c => g.appendChild(c));
  });
  document.querySelectorAll('.sort-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.by === by));
}
function setProc(p) {
  document.querySelectorAll('.card').forEach(c => {
    c.style.display = (p === 'all' || c.dataset.proc === p) ? '' : 'none';
  });
  document.querySelectorAll('.proc-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.proc === p));
}
function pauseAllExcept(target) {
  document.querySelectorAll('video').forEach(v => {
    if (v !== target && !v.paused) v.pause();
  });
}
document.addEventListener('play', e => pauseAllExcept(e.target), true);
"""


def render(rows: list[dict], root: Path) -> str:
    total = len(rows)
    procs = sorted({r["proc"] for r in rows})
    buckets: dict[str, list[dict]] = {b: [] for b in BUCKET_ORDER}
    for r in rows:
        buckets.setdefault(r["bucket"], []).append(r)
    for b in buckets.values():
        b.sort(key=lambda r: r["distance_to_goal"])

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append(f"<title>VLFM episodes — {root.name}</title>")
    parts.append(f"<style>{CSS}</style></head><body>")
    parts.append(f"<h1>VLFM episode videos — {html.escape(root.name)}</h1>")
    parts.append(
        f"<div class='summary'>built {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"&middot; {total} videos &middot; {len(procs)} proc(s): "
        f"{', '.join(procs)} &middot; reload page for fresh state</div>"
    )

    parts.append("<div class='controls'>")
    parts.append("<span style='color:#888;font-size:12px;'>sort:</span>")
    for by, label in [
        ("distance_to_goal", "distance"),
        ("episode", "episode"),
        ("spl", "spl"),
    ]:
        active = "active" if by == "distance_to_goal" else ""
        parts.append(
            f"<button class='sort-btn {active}' data-by='{by}' "
            f"onclick=\"setSort('{by}')\">{label}</button>"
        )
    parts.append("<span style='color:#888;font-size:12px;margin-left:8px;'>proc:</span>")
    parts.append(
        "<button class='proc-btn active' data-proc='all' "
        "onclick=\"setProc('all')\">all</button>"
    )
    for p in procs:
        parts.append(
            f"<button class='proc-btn' data-proc='{p}' "
            f"onclick=\"setProc('{p}')\">{p}</button>"
        )
    parts.append("</div>")

    for b in BUCKET_ORDER:
        items = buckets.get(b) or []
        if not items:
            continue
        parts.append(f"<h2 class='bucket'>{html.escape(BUCKET_LABEL[b])} &middot; {len(items)}</h2>")
        parts.append("<div class='grid'>")
        for r in items:
            det_cls = "detected" if r["target_detected"] >= 0.5 else "no-detected"
            stop_cls = "stopped" if r["stop_called"] >= 0.5 else "no-stopped"
            det_text = "detected" if r["target_detected"] >= 0.5 else "no detect"
            stop_text = "stopped" if r["stop_called"] >= 0.5 else "no stop"
            parts.append(
                f"<div class='card' data-proc='{r['proc']}' "
                f"data-episode='{r['episode']}' "
                f"data-distance_to_goal='{r['distance_to_goal']:.3f}' "
                f"data-spl='{r['spl']:.3f}'>"
            )
            parts.append(
                f"<video controls preload='none' "
                f"src='{html.escape(r['rel_path'])}'></video>"
            )
            parts.append("<div class='meta'>")
            parts.append(
                f"<span class='tag proc'>{r['proc']}</span>"
                f"<span class='tag {det_cls}'>{det_text}</span>"
                f"<span class='tag {stop_cls}'>{stop_text}</span>"
            )
            parts.append(
                f"<br>ep <b>{r['episode']}</b> "
                f"&middot; dist <span class='dist'>{r['distance_to_goal']:.2f} m</span> "
                f"&middot; spl <span class='spl'>{r['spl']:.2f}</span> "
                f"&middot; soft_spl {r['soft_spl']:.2f} "
                f"&middot; {r['size_mb']:.1f} MB"
            )
            parts.append("</div></div>")
        parts.append("</div>")

    if not rows:
        parts.append("<div class='empty'>no videos found yet</div>")

    parts.append(f"<script>{JS}</script></body></html>")
    return "\n".join(parts)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"not a dir: {root}", file=sys.stderr)
        return 2
    rows = collect(root)
    out = root / "index.html"
    out.write_text(render(rows, root), encoding="utf-8")
    print(f"wrote {out}  ({len(rows)} videos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
