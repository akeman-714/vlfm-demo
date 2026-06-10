#!/usr/bin/env python3
"""Web UI for ObjectNav demo and navigation-test runs.

Flow:
  run mode -> demo eval split/env -> live RGB/BEV frames -> saved mp4
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from semantic_goal_head import DEFAULT_BASE_URL, DEFAULT_MODEL, GoalResolution, resolve_goal


REPO = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO / "outputs" / "semantic_cat_web_runs"
ALLOWED_LABELS = ("cat", "toilet")
LABEL_TO_SPLIT = {
    "cat": "cat_demo",
    "toilet": "toilet_demo",
}
RUN_MODES: dict[str, dict[str, Any]] = {
    "semantic_query": {
        "name": "Semantic Query",
        "label": "semantic",
        "split": None,
        "stage": "semantic",
        "env": {},
        "semantic": True,
    },
    "find_cat": {
        "name": "Find Cat",
        "label": "cat",
        "split": "cat_demo",
        "stage": "finding",
        "env": {},
    },
    "global_home_40": {
        "name": "Global Home 40",
        "label": "home@40",
        "split": "cat_demo",
        "stage": "global-home-40",
        "env": {
            "VLFM_GLOBAL_NAV": "1",
            "VLFM_NAV_DEBUG_GOAL": "0,0",
            "VLFM_NAV_DEBUG_AFTER": "40",
            "VLFM_NAV_DEBUG_CONSERVATIVE": "1",
            "VLFM_NAV_DEBUG_LOG": "1",
        },
    },
    "global_home_100": {
        "name": "Global Home 100",
        "label": "home@100",
        "split": "cat_demo",
        "stage": "global-home-100",
        "env": {
            "VLFM_GLOBAL_NAV": "1",
            "VLFM_NAV_DEBUG_GOAL": "0,0",
            "VLFM_NAV_DEBUG_AFTER": "100",
            "VLFM_NAV_DEBUG_CONSERVATIVE": "1",
            "VLFM_NAV_DEBUG_LOG": "1",
        },
    },
    "object_memory_cat": {
        "name": "Object Memory Cat",
        "label": "memory-cat",
        "split": "cat_demo",
        "stage": "object-memory",
        "env": {
            "VLFM_GLOBAL_NAV": "1",
            "VLFM_OBJECT_MEMORY_PATH": str(REPO / "data" / "object_memory" / "module2_demo" / "cat.json"),
            "VLFM_NAV_DEBUG_LOG": "1",
        },
    },
    "persistent_memory_cat_pair": {
        "name": "Persistent Map + Memory Cat Pair",
        "label": "map+memory-cat",
        "split": "cat_demo",
        "stage": "persistent-memory-pair",
        "env": {
            "VLFM_GLOBAL_NAV": "1",
            "VLFM_NAV_DEBUG_LOG": "1",
            "VLFM_MEMORY_NAV_CONSERVATIVE": "0",
        },
        "paired_persistent": True,
    },
    "multi_goal_cat": {
        "name": "Multi-goal: origin -> fridge -> cat -> origin",
        "label": "multigoal",
        "split": "cat_demo",
        "stage": "multi-goal",
        "env": {
            "VLFM_GLOBAL_NAV": "1",
            "VLFM_NAV_DEBUG_LOG": "1",
            "VLFM_MEMORY_NAV_CONSERVATIVE": "0",
        },
        "paired_persistent": True,
        # Pass 1 = single-goal find-cat (builds obstacle map + cat memory).
        # Pass 2 = ordered multi-goal plan. Object tokens must match the detector
        # vocabulary (English MP3D/COCO names), so use "refrigerator" not "冰箱";
        # change the sequence freely (origin = episodic start).
        "pass2_env": {"VLFM_GOAL_SEQUENCE": "origin, refrigerator, cat, origin"},
        "pass2_reason": "Pass 2/2: multi-goal (origin -> refrigerator[explore] -> cat[memory A*] -> origin)",
    },
}


@dataclass
class RunState:
    run_id: str
    request_text: str
    run_mode: str = "find_cat"
    status: str = "starting"
    stage: str = "queued"
    label: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    run_dir: str = ""
    live_dir: str = ""
    live_bev_dir: str = ""
    video_dir: str = ""
    tb_dir: str = ""
    log_path: str = ""
    video_path: Optional[str] = None
    live_video_path: Optional[str] = None
    latest_frame: Optional[str] = None
    latest_bev_frame: Optional[str] = None


app = Flask(__name__, static_folder=None)
state_lock = threading.Lock()
runs: dict[str, RunState] = {}
active_run_id: Optional[str] = None


def _tail(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.exists():
        return None
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _safe_run(run_id: str) -> RunState:
    with state_lock:
        if run_id not in runs:
            raise KeyError(run_id)
        return runs[run_id]


def _frame_token(path_str: Optional[str]) -> Optional[str]:
    """Stable per-frame id (mtime in ms) so the browser only refetches when a frame changes.

    Without this the page cache-busts every poll and re-downloads the *same* frame ~2x
    (poll 0.8s < generation ~1.5s). The mtime also differs across the two-pass modes, where
    filenames restart at env0_step0000 in a new stage dir.
    """
    if not path_str:
        return None
    try:
        return str(int(os.path.getmtime(path_str) * 1000))
    except OSError:
        return None


def _serialize(run: RunState) -> dict[str, Any]:
    data = asdict(run)
    data["elapsed_sec"] = round((run.finished_at or time.time()) - run.started_at, 1)
    data["latest_frame_url"] = f"/api/runs/{run.run_id}/latest-frame" if run.latest_frame else None
    data["latest_bev_frame_url"] = f"/api/runs/{run.run_id}/latest-bev-frame" if run.latest_bev_frame else None
    data["latest_frame_token"] = _frame_token(run.latest_frame)
    data["latest_bev_frame_token"] = _frame_token(run.latest_bev_frame)
    data["video_url"] = f"/api/runs/{run.run_id}/video" if run.video_path else None
    data["live_video_url"] = f"/api/runs/{run.run_id}/live-video" if run.live_video_path else None
    data["log_tail"] = _tail(Path(run.log_path), 100) if run.log_path else ""
    return data


def _set_run(run_id: str, **changes: Any) -> None:
    with state_lock:
        run = runs[run_id]
        for key, value in changes.items():
            setattr(run, key, value)


def _watch_outputs(run_id: str, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        with state_lock:
            run = runs.get(run_id)
            if run is None:
                return
            live_dir = Path(run.live_dir)
            live_bev_dir = Path(run.live_bev_dir)
            video_dir = Path(run.video_dir)

        latest_frame = _latest_file(live_dir, "*.png")
        latest_bev_frame = _latest_file(live_bev_dir, "*.png")
        latest_video = _latest_file(video_dir, "*.mp4")
        changes: dict[str, Any] = {}
        if latest_frame is not None:
            changes["latest_frame"] = str(latest_frame)
        if latest_bev_frame is not None:
            changes["latest_bev_frame"] = str(latest_bev_frame)
        if latest_video is not None:
            changes["video_path"] = str(latest_video)
        if changes:
            _set_run(run_id, **changes)
        time.sleep(0.5)


def _build_live_rgb_video(live_dir: Path, out_path: Path) -> Optional[Path]:
    frames = sorted(live_dir.glob("env0_step*.png"))
    if not frames:
        return None
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(out_path, fps=10, macro_block_size=16) as writer:
            for frame in frames:
                writer.append_data(imageio.imread(frame))
    except Exception:
        return None
    return out_path if out_path.exists() else None


def _encode_preview(path: Path, max_dim: int, quality: int) -> Optional[bytes]:
    """Downscale + JPEG-encode a dumped frame for the live preview.

    Live frames are 640x480 RGB / variable-size BEV PNGs (~160KB / ~60KB). Shipping those
    raw over the SSH tunnel is what makes the stream fall behind generation, so we shrink
    them to a JPEG (~15-20KB) on the way out. The on-disk PNGs and the archived mp4 keep
    full resolution. Returns None (caller falls back to the raw PNG) if cv2/encode fails or
    the file is mid-write.
    """
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = max_dim / float(max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return buf.tobytes()
    except Exception:
        return None


def _serve_preview_frame(path_str: Optional[str], max_dim: int, quality: int) -> Response:
    frame = Path(path_str or "")
    if not frame.exists():
        return Response(status=204)
    data = _encode_preview(frame, max_dim=max_dim, quality=quality)
    if data is None:
        return send_file(frame, mimetype="image/png", max_age=0)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


def _resolve_semantic_goal(text: str, api_key_override: str = "") -> GoalResolution:
    api_key = (
        api_key_override.strip()
        or os.environ.get("BAILIAN_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Missing API key. Enter one in the page or set BAILIAN_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY."
        )

    return resolve_goal(
        text,
        api_key=api_key,
        base_url=os.environ.get("BAILIAN_BASE_URL", DEFAULT_BASE_URL),
        model=os.environ.get("BAILIAN_MODEL", DEFAULT_MODEL),
        allowed_labels=ALLOWED_LABELS,
        timeout=float(os.environ.get("BAILIAN_TIMEOUT", "20")),
    )


def _run_eval_process(
    run_id: str,
    split: str,
    stage_name: str,
    env_overrides: dict[str, Any],
    stage_root: Path,
) -> tuple[int, Optional[Path], Optional[Path]]:
    run = _safe_run(run_id)
    video_dir = stage_root / "video"
    tb_dir = stage_root / "tb"
    live_dir = stage_root / "live_rgb"
    live_bev_dir = stage_root / "live_bev"
    log_path = stage_root / "vlfm.log"
    for directory in (stage_root, video_dir, tb_dir, live_dir, live_bev_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _set_run(
        run_id,
        stage=stage_name,
        live_dir=str(live_dir),
        live_bev_dir=str(live_bev_dir),
        video_dir=str(video_dir),
        tb_dir=str(tb_dir),
        log_path=str(log_path),
        video_path=None,
        live_video_path=None,
        latest_frame=None,
        latest_bev_frame=None,
    )

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0,7"),
            "SPLIT": split,
            "N_EP": "1",
            "VIDEO_DIR": str(video_dir),
            "TB_DIR": str(tb_dir),
            "LOG": str(log_path),
            "VLFM_DUMP_RGB_DIR": str(live_dir),
            "VLFM_DUMP_BEV_DIR": str(live_bev_dir),
        }
    )
    env.update({str(k): str(v) for k, v in env_overrides.items()})

    proc = subprocess.Popen(
        ["bash", "scripts/cat_demo/eval_cat_demo.sh"],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    web_log = stage_root / "launcher.out"
    with web_log.open("w", encoding="utf-8") as f:
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()
    exit_code = proc.wait()

    latest_frame = _latest_file(live_dir, "*.png")
    latest_bev_frame = _latest_file(live_bev_dir, "*.png")
    latest_video = _latest_file(video_dir, "*.mp4")
    live_video = _build_live_rgb_video(live_dir, stage_root / "live_rgb_preview.mp4")
    changes: dict[str, Any] = {}
    if latest_frame is not None:
        changes["latest_frame"] = str(latest_frame)
    if latest_bev_frame is not None:
        changes["latest_bev_frame"] = str(latest_bev_frame)
    if latest_video is not None:
        changes["video_path"] = str(latest_video)
    if live_video is not None:
        changes["live_video_path"] = str(live_video)
    if changes:
        _set_run(run_id, **changes)

    if exit_code != 0:
        log_tail = _tail(log_path, 40) or _tail(web_log, 40)
        raise RuntimeError(f"VLFM eval exited with code {exit_code} during {stage_name}.\n{log_tail}")
    if latest_video is None:
        raise RuntimeError(f"VLFM eval finished during {stage_name} but no mp4 was written.")

    return exit_code, latest_video, live_video


def _run_eval(run_id: str, text: str, run_mode: str, api_key: str = "") -> None:
    global active_run_id

    stop_watch = threading.Event()
    watcher: Optional[threading.Thread] = None
    try:
        mode_config = RUN_MODES.get(run_mode)
        if mode_config is None:
            modes = ", ".join(sorted(RUN_MODES))
            raise RuntimeError(f"Unknown run mode {run_mode!r}; supported modes: {modes}.")

        if mode_config.get("semantic"):
            _set_run(run_id, status="running", stage="semantic", label="semantic", reason="Semantic Query")
            semantic = _resolve_semantic_goal(text, api_key)
            split = LABEL_TO_SPLIT.get(semantic.label or "")
            if split is None:
                labels = ", ".join(sorted(LABEL_TO_SPLIT))
                raise RuntimeError(f"Resolved label is {semantic.label!r}; supported demo labels: {labels}.")
            _set_run(
                run_id,
                label=semantic.label,
                confidence=semantic.confidence,
                reason=f"Semantic Query -> {semantic.label}: {semantic.reason}",
                stage="finding",
            )
        else:
            split = str(mode_config["split"])
            _set_run(
                run_id,
                status="running",
                stage=str(mode_config["stage"]),
                label=str(mode_config["label"]),
                confidence=1.0,
                reason=str(mode_config["name"]),
            )

        run = _safe_run(run_id)
        run_dir = Path(run.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        stop_watch.clear()
        watcher = threading.Thread(target=_watch_outputs, args=(run_id, stop_watch), daemon=True)
        watcher.start()

        if mode_config.get("paired_persistent"):
            asset_dir = run_dir / "assets"
            map_path = asset_dir / "cat_map.npz"
            memory_path = asset_dir / "cat.json"
            for stale_path in (map_path, memory_path, Path(str(map_path) + ".lock"), Path(str(memory_path) + ".lock")):
                if stale_path.exists():
                    stale_path.unlink()

            paired_env = dict(mode_config["env"])
            paired_env.update(
                {
                    "VLFM_PERSIST_MAP_PATH": str(map_path),
                    "VLFM_OBJECT_MEMORY_PATH": str(memory_path),
                }
            )

            _set_run(run_id, reason="Pass 1/2: find cat, save obstacle map and cat memory")
            _run_eval_process(
                run_id,
                split,
                "pass-1-build-map-memory",
                paired_env,
                run_dir / "pass1_build",
            )
            if not map_path.exists():
                raise RuntimeError(f"Pass 1 finished but persistent map was not written: {map_path}")
            if not memory_path.exists():
                raise RuntimeError(f"Pass 1 finished but object memory was not written: {memory_path}")

            pass2_env = dict(paired_env)
            pass2_env.update(mode_config.get("pass2_env", {}))
            _set_run(
                run_id,
                reason=str(mode_config.get("pass2_reason", "Pass 2/2: load obstacle map + cat memory, then A* to memory")),
            )
            exit_code, _, _ = _run_eval_process(
                run_id,
                split,
                "pass-2-load-and-a-star",
                pass2_env,
                run_dir / "pass2_recall",
            )
        else:
            stage_name = "finding" if mode_config.get("semantic") else str(mode_config["stage"])
            exit_code, _, _ = _run_eval_process(
                run_id,
                split,
                stage_name,
                dict(mode_config["env"]),
                run_dir,
            )

        _set_run(run_id, status="complete", stage="complete", exit_code=exit_code, finished_at=time.time())
    except Exception as exc:
        _set_run(
            run_id,
            status="error",
            stage="error",
            error=str(exc),
            finished_at=time.time(),
        )
    finally:
        stop_watch.set()
        if watcher is not None:
            watcher.join(timeout=2)
        with state_lock:
            if active_run_id == run_id:
                active_run_id = None


@app.get("/")
def index() -> Response:
    return send_file(REPO / "scripts" / "cat_demo" / "web_static" / "index.html")


@app.post("/api/runs")
def create_run() -> Response:
    global active_run_id
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    api_key = str(payload.get("api_key", "")).strip()
    run_mode = str(payload.get("run_mode", "find_cat")).strip() or "find_cat"
    if run_mode not in RUN_MODES:
        return jsonify({"error": f"unknown run mode: {run_mode}"}), 400
    if RUN_MODES[run_mode].get("semantic") and not text:
        return jsonify({"error": "empty semantic query"}), 400

    with state_lock:
        if active_run_id is not None:
            return jsonify({"error": "a run is already active", "run_id": active_run_id}), 409
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = RUN_ROOT / run_id
        run = RunState(
            run_id=run_id,
            request_text=text or RUN_MODES[run_mode]["name"],
            run_mode=run_mode,
            run_dir=str(run_dir),
            live_dir=str(run_dir / "live_rgb"),
            live_bev_dir=str(run_dir / "live_bev"),
            video_dir=str(run_dir / "video"),
            tb_dir=str(run_dir / "tb"),
            log_path=str(run_dir / "vlfm.log"),
        )
        runs[run_id] = run
        active_run_id = run_id

    thread = threading.Thread(target=_run_eval, args=(run_id, text, run_mode, api_key), daemon=True)
    thread.start()
    return jsonify(_serialize(run))


@app.get("/api/runs/<run_id>")
def get_run(run_id: str) -> Response:
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            return jsonify({"error": "run not found"}), 404
        data = _serialize(run)
    return jsonify(data)


@app.get("/api/runs/<run_id>/latest-frame")
def latest_frame(run_id: str) -> Response:
    run = _safe_run(run_id)
    return _serve_preview_frame(run.latest_frame, max_dim=480, quality=72)


@app.get("/api/runs/<run_id>/latest-bev-frame")
def latest_bev_frame(run_id: str) -> Response:
    run = _safe_run(run_id)
    return _serve_preview_frame(run.latest_bev_frame, max_dim=480, quality=80)


@app.get("/api/runs/<run_id>/video")
def video(run_id: str) -> Response:
    run = _safe_run(run_id)
    video_path = Path(run.video_path or "")
    if not video_path.exists():
        return Response(status=204)
    return send_file(video_path, mimetype="video/mp4", as_attachment=False, max_age=0)


@app.get("/api/runs/<run_id>/live-video")
def live_video(run_id: str) -> Response:
    run = _safe_run(run_id)
    video_path = Path(run.live_video_path or "")
    if not video_path.exists():
        return Response(status=204)
    return send_file(video_path, mimetype="video/mp4", as_attachment=False, max_age=0)


@app.get("/static/<path:name>")
def static_file(name: str) -> Response:
    return send_from_directory(REPO / "scripts" / "cat_demo" / "web_static", name)


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("SEMANTIC_CAT_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("SEMANTIC_CAT_WEB_PORT", "7860"))
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
