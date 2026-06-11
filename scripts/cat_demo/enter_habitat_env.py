#!/usr/bin/env python3
"""Small browser viewer for manually inspecting the merged cat Habitat scene.

This runs Habitat-Sim headlessly on the server GPU and serves the current RGB
camera view over HTTP. Open the printed URL through an SSH tunnel, then use the
keyboard in the browser:

  W/S: move forward/back       A/D: turn left/right
  Q/E: strafe left/right       Z/C: look down/up
  R: reset to the selected start pose
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
from flask import Flask, Response, jsonify, request
import habitat_sim
import imageio.v2 as imageio
import numpy as np
import quaternion


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GLB = REPO / "data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
DEFAULT_SCENE_DATASET = REPO / "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
DEFAULT_EPISODE = REPO / "data/datasets/objectnav/hm3d/v1/cat_demo/content/TEEsavR23oF.json.gz"


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Habitat Cat Scene Viewer</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
    body { margin: 0; background: #111; color: #eee; display: grid; min-height: 100vh; grid-template-rows: auto 1fr auto; }
    header, footer { padding: 10px 14px; background: #1d1d1d; display: flex; gap: 18px; flex-wrap: wrap; align-items: center; }
    main { display: grid; place-items: center; padding: 12px; }
    img { max-width: min(100%, 1200px); max-height: calc(100vh - 140px); object-fit: contain; background: #000; border: 1px solid #333; }
    button { background: #2b2b2b; color: #eee; border: 1px solid #555; border-radius: 6px; padding: 7px 10px; cursor: pointer; }
    button:hover { background: #3a3a3a; }
    .muted { color: #aaa; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
  </style>
</head>
<body>
  <header>
    <strong>Habitat Cat Scene Viewer</strong>
    <span id="pose" class="mono muted">loading...</span>
    <span id="gpu" class="mono muted"></span>
  </header>
  <main>
    <img id="frame" alt="Habitat RGB frame" src="/frame.jpg">
  </main>
  <footer>
    <button data-action="forward">W forward</button>
    <button data-action="back">S back</button>
    <button data-action="left">A turn left</button>
    <button data-action="right">D turn right</button>
    <button data-action="strafe_left">Q strafe left</button>
    <button data-action="strafe_right">E strafe right</button>
    <button data-action="look_down">Z look down</button>
    <button data-action="look_up">C look up</button>
    <button data-action="reset">R reset</button>
    <button id="save">save frame</button>
    <span class="muted">Click the page once, then use keyboard.</span>
  </footer>
  <script>
    const frame = document.getElementById("frame");
    const pose = document.getElementById("pose");
    const gpu = document.getElementById("gpu");
    let busy = false;
    let frameUrl = "";

    function updateState(st) {
      pose.textContent = `pos=${st.position.map(v => v.toFixed(3)).join(", ")} yaw=${st.yaw_deg.toFixed(1)} pitch=${st.pitch_deg.toFixed(1)}`;
      gpu.textContent = `CUDA_VISIBLE_DEVICES=${st.cuda_visible_devices || ""}`;
    }

    async function refresh() {
      const st = await fetch("/api/state").then(r => r.json());
      updateState(st);
      frame.src = `/frame.jpg?t=${Date.now()}`;
    }

    async function act(action) {
      if (busy) return;
      busy = true;
      try {
        const res = await fetch("/api/action-frame", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({action})
        });
        const stateHeader = res.headers.get("X-Viewer-State");
        if (stateHeader) updateState(JSON.parse(decodeURIComponent(stateHeader)));
        const blob = await res.blob();
        if (frameUrl) URL.revokeObjectURL(frameUrl);
        frameUrl = URL.createObjectURL(blob);
        frame.src = frameUrl;
      } finally {
        busy = false;
      }
    }

    document.querySelectorAll("button[data-action]").forEach(btn => {
      btn.addEventListener("click", () => act(btn.dataset.action));
    });
    document.getElementById("save").addEventListener("click", async () => {
      const st = await fetch("/api/save", {method: "POST"}).then(r => r.json());
      alert(`saved: ${st.path}`);
    });
    document.addEventListener("keydown", ev => {
      const keymap = {
        w: "forward", s: "back", a: "left", d: "right",
        q: "strafe_left", e: "strafe_right", z: "look_down",
        c: "look_up", r: "reset"
      };
      const action = keymap[ev.key.toLowerCase()];
      if (action) {
        ev.preventDefault();
        act(action);
      }
    });
    refresh();
  </script>
</body>
</html>
"""


def yaw_to_quat(yaw_rad: float) -> np.quaternion:
    return quaternion.from_rotation_vector([0.0, yaw_rad, 0.0])


def load_episode(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return data


def initial_pose(args: argparse.Namespace) -> tuple[np.ndarray, float]:
    if args.xyz:
        vals = [float(v) for v in args.xyz.split(",")]
        if len(vals) != 3:
            raise ValueError("--xyz must be 'x,y,z'")
        return np.array(vals, dtype=np.float64), math.radians(args.yaw_deg)

    data = load_episode(args.episode)
    ep = data["episodes"][0]
    if args.pose == "episode":
        pos = np.array(ep["start_position"], dtype=np.float64)
        rot = ep["start_rotation"]
        q = np.quaternion(rot[3], rot[0], rot[1], rot[2])
        _, yaw, _ = quaternion.as_euler_angles(q)
        return pos, float(yaw)

    key = f"{args.glb.name}_{ep['object_category']}"
    goal = np.array(data["goals_by_category"][key][0]["position"], dtype=np.float64)
    idx = int(args.pose.removeprefix("viewpoint"))
    view_points = data["goals_by_category"][key][0]["view_points"]
    pos = np.array(view_points[idx]["agent_state"]["position"], dtype=np.float64)
    eye = pos + np.array([0.0, args.camera_height, 0.0])
    vec = goal - eye
    yaw = math.atan2(vec[0], -vec[2])
    return pos, yaw


class Viewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.pitch = math.radians(args.pitch_deg)
        self.start_pos, self.start_yaw = initial_pose(args)
        self.pos = self.start_pos.copy()
        self.yaw = self.start_yaw
        self.sim = self._make_sim()
        self.agent = self.sim.get_agent(0)
        self._apply_state()

    def _make_sim(self) -> habitat_sim.Simulator:
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = str(self.args.glb)
        backend_cfg.scene_dataset_config_file = str(self.args.scene_dataset)
        backend_cfg.enable_physics = False

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [self.args.height, self.args.width]
        rgb_spec.position = [0.0, self.args.camera_height, 0.0]
        rgb_spec.hfov = self.args.hfov

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = self.args.camera_height
        agent_cfg.radius = self.args.agent_radius
        agent_cfg.sensor_specifications = [rgb_spec]
        return habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))

    def close(self) -> None:
        self.sim.close()

    def _rotation(self) -> np.quaternion:
        return (
            quaternion.from_rotation_vector([self.pitch, 0.0, 0.0])
            * yaw_to_quat(self.yaw)
        )

    def _apply_state(self) -> None:
        st = habitat_sim.AgentState()
        st.position = self.pos.astype(np.float32)
        st.rotation = self._rotation()
        self.agent.set_state(st)

    def _try_move(self, delta: np.ndarray) -> None:
        desired = self.pos + delta
        if self.args.no_navmesh:
            self.pos = desired
            return
        stepped = np.asarray(
            self.sim.pathfinder.try_step(
                self.pos.astype(np.float32),
                desired.astype(np.float32),
            ),
            dtype=np.float64,
        )
        if np.all(np.isfinite(stepped)):
            self.pos = stepped

    def action(self, name: str) -> None:
        with self.lock:
            if name == "reset":
                self.pos = self.start_pos.copy()
                self.yaw = self.start_yaw
                self.pitch = math.radians(self.args.pitch_deg)
            elif name in {"left", "right"}:
                sign = 1.0 if name == "left" else -1.0
                self.yaw += sign * math.radians(self.args.turn_deg)
            elif name in {"look_up", "look_down"}:
                sign = 1.0 if name == "look_up" else -1.0
                self.pitch = float(np.clip(
                    self.pitch + sign * math.radians(self.args.look_deg),
                    math.radians(-65),
                    math.radians(65),
                ))
            elif name in {"forward", "back", "strafe_left", "strafe_right"}:
                forward = np.array([math.sin(self.yaw), 0.0, -math.cos(self.yaw)])
                right = np.array([math.cos(self.yaw), 0.0, math.sin(self.yaw)])
                direction = {
                    "forward": forward,
                    "back": -forward,
                    "strafe_left": -right,
                    "strafe_right": right,
                }[name]
                self._try_move(direction * self.args.step_size)
            self._apply_state()

    def frame(self) -> np.ndarray:
        with self.lock:
            obs = self.sim.get_sensor_observations()
            rgb = np.asarray(obs["rgb"])[..., :3].copy()
            label = (
                f"pos=({self.pos[0]:.2f},{self.pos[1]:.2f},{self.pos[2]:.2f}) "
                f"yaw={math.degrees(self.yaw):+.1f} pitch={math.degrees(self.pitch):+.1f}"
            )
            cv2.putText(
                rgb,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                rgb,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            return rgb

    def jpeg(self) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(self.frame(), cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("failed to encode frame")
        return encoded.tobytes()

    def state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "position": self.pos.tolist(),
                "yaw_deg": math.degrees(self.yaw),
                "pitch_deg": math.degrees(self.pitch),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }


def build_app(viewer: Viewer) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        return Response(INDEX_HTML, mimetype="text/html")

    @app.get("/frame.jpg")
    def frame() -> Response:
        return Response(viewer.jpeg(), mimetype="image/jpeg")

    @app.get("/api/state")
    def state() -> Response:
        return jsonify(viewer.state())

    @app.post("/api/action")
    def action() -> Response:
        data = request.get_json(silent=True) or {}
        viewer.action(str(data.get("action", "")))
        return jsonify(viewer.state())

    @app.post("/api/action-frame")
    def action_frame() -> Response:
        data = request.get_json(silent=True) or {}
        viewer.action(str(data.get("action", "")))
        state_json = json.dumps(viewer.state(), separators=(",", ":"))
        resp = Response(viewer.jpeg(), mimetype="image/jpeg")
        resp.headers["X-Viewer-State"] = quote(state_json)
        return resp

    @app.post("/api/save")
    def save() -> Response:
        out_dir = viewer.args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = len(list(out_dir.glob("frame_*.jpg")))
        out = out_dir / f"frame_{idx:04d}.jpg"
        imageio.imwrite(out, viewer.frame())
        return jsonify({"path": str(out)})

    return app


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glb", type=Path, default=DEFAULT_GLB)
    ap.add_argument("--scene-dataset", type=Path, default=DEFAULT_SCENE_DATASET)
    ap.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    ap.add_argument(
        "--pose",
        default="episode",
        help="episode, viewpoint0, viewpoint1, ... (default: episode)",
    )
    ap.add_argument("--xyz", help="manual start position as x,y,z")
    ap.add_argument("--yaw-deg", type=float, default=0.0)
    ap.add_argument("--pitch-deg", type=float, default=0.0)
    ap.add_argument("--host", default=os.environ.get("HABITAT_VIEWER_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("HABITAT_VIEWER_PORT", "7862")))
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--camera-height", type=float, default=0.88)
    ap.add_argument("--agent-radius", type=float, default=0.18)
    ap.add_argument("--step-size", type=float, default=0.25)
    ap.add_argument("--turn-deg", type=float, default=15.0)
    ap.add_argument("--look-deg", type=float, default=10.0)
    ap.add_argument("--jpeg-quality", type=int, default=40)
    ap.add_argument("--no-navmesh", action="store_true", help="free-fly instead of pathfinder.try_step")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "habitat_viewer_frames")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    viewer = Viewer(args)
    app = build_app(viewer)
    print(">>> Habitat Cat Scene Viewer")
    print(f">>> scene : {args.glb}")
    print(f">>> pose  : {args.pose}")
    print(f">>> URL   : http://{args.host}:{args.port}")
    print(f">>> CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    try:
        app.run(host=args.host, port=args.port, threaded=False)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
