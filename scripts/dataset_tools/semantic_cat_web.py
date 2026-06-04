#!/usr/bin/env python3
"""Web UI for natural-language ObjectNav demos.

Flow:
  user text -> Bailian semantic label -> demo eval split -> live RGB frames -> saved mp4
"""
from __future__ import annotations

import json
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


@dataclass
class RunState:
    run_id: str
    request_text: str
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
    video_dir: str = ""
    tb_dir: str = ""
    log_path: str = ""
    video_path: Optional[str] = None
    live_video_path: Optional[str] = None
    latest_frame: Optional[str] = None


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


def _serialize(run: RunState) -> dict[str, Any]:
    data = asdict(run)
    data["elapsed_sec"] = round((run.finished_at or time.time()) - run.started_at, 1)
    data["latest_frame_url"] = f"/api/runs/{run.run_id}/latest-frame" if run.latest_frame else None
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
            video_dir = Path(run.video_dir)

        latest_frame = _latest_file(live_dir, "*.png")
        latest_video = _latest_file(video_dir, "*.mp4")
        changes: dict[str, Any] = {}
        if latest_frame is not None:
            changes["latest_frame"] = str(latest_frame)
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


def _run_eval(run_id: str, text: str, api_key: str = "") -> None:
    global active_run_id

    stop_watch = threading.Event()
    watcher: Optional[threading.Thread] = None
    try:
        _set_run(run_id, status="running", stage="semantic")
        semantic = _resolve_semantic_goal(text, api_key)
        _set_run(run_id, label=semantic.label, confidence=semantic.confidence, reason=semantic.reason)
        split = LABEL_TO_SPLIT.get(semantic.label or "")
        if split is None:
            labels = ", ".join(sorted(LABEL_TO_SPLIT))
            raise RuntimeError(f"Resolved label is {semantic.label!r}; supported demo labels: {labels}.")

        run = _safe_run(run_id)
        run_dir = Path(run.run_dir)
        video_dir = Path(run.video_dir)
        tb_dir = Path(run.tb_dir)
        live_dir = Path(run.live_dir)
        for directory in (run_dir, video_dir, tb_dir, live_dir):
            directory.mkdir(parents=True, exist_ok=True)

        _set_run(run_id, stage="finding")
        stop_watch.clear()
        watcher = threading.Thread(target=_watch_outputs, args=(run_id, stop_watch), daemon=True)
        watcher.start()

        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0,7"),
                "SPLIT": split,
                "N_EP": "1",
                "VIDEO_DIR": str(video_dir),
                "TB_DIR": str(tb_dir),
                "LOG": str(Path(run.log_path)),
                "VLFM_DUMP_RGB_DIR": str(live_dir),
            }
        )

        proc = subprocess.Popen(
            ["bash", "scripts/dataset_tools/eval_cat_demo.sh"],
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        web_log = run_dir / "launcher.out"
        with web_log.open("w", encoding="utf-8") as f:
            assert proc.stdout is not None
            for line in proc.stdout:
                f.write(line)
                f.flush()
        exit_code = proc.wait()

        latest_frame = _latest_file(live_dir, "*.png")
        latest_video = _latest_file(video_dir, "*.mp4")
        live_video = _build_live_rgb_video(live_dir, run_dir / "live_rgb_preview.mp4")
        if latest_frame is not None:
            _set_run(run_id, latest_frame=str(latest_frame))
        if latest_video is not None:
            _set_run(run_id, video_path=str(latest_video))
        if live_video is not None:
            _set_run(run_id, live_video_path=str(live_video))

        if exit_code != 0:
            log_tail = _tail(Path(run.log_path), 40) or _tail(web_log, 40)
            raise RuntimeError(f"VLFM eval exited with code {exit_code}.\n{log_tail}")
        if latest_video is None:
            raise RuntimeError("VLFM eval finished but no mp4 was written.")

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
    return send_file(REPO / "scripts" / "dataset_tools" / "semantic_cat_web_static" / "index.html")


@app.post("/api/runs")
def create_run() -> Response:
    global active_run_id
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    api_key = str(payload.get("api_key", "")).strip()
    if not text:
        return jsonify({"error": "empty request text"}), 400

    with state_lock:
        if active_run_id is not None:
            return jsonify({"error": "a run is already active", "run_id": active_run_id}), 409
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = RUN_ROOT / run_id
        run = RunState(
            run_id=run_id,
            request_text=text,
            run_dir=str(run_dir),
            live_dir=str(run_dir / "live_rgb"),
            video_dir=str(run_dir / "video"),
            tb_dir=str(run_dir / "tb"),
            log_path=str(run_dir / "vlfm.log"),
        )
        runs[run_id] = run
        active_run_id = run_id

    thread = threading.Thread(target=_run_eval, args=(run_id, text, api_key), daemon=True)
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
    frame = Path(run.latest_frame or "")
    if not frame.exists():
        return Response(status=204)
    return send_file(frame, mimetype="image/png", max_age=0)


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
    return send_from_directory(REPO / "scripts" / "dataset_tools" / "semantic_cat_web_static", name)


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("SEMANTIC_CAT_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("SEMANTIC_CAT_WEB_PORT", "7860"))
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
