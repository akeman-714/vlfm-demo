# 06 · 仿真环境：Habitat 本体、接口、通信与 RGBD 数据流

> 这一章把"环境怎么生成 RGB-D / 怎么塞进策略 / 策略怎么把动作回塞给环境"讲透。
>
> VLFM 支持 3 套环境，但**真正训练/评测的主路径是 Habitat**（其余两套保留接口一致）：
>
> | 环境 | 入口脚本 | 用途 | 仿真器 |
> | --- | --- | --- | --- |
> | **Habitat ObjectNav** | `python -m vlfm.run` (`vlfm/run.py`) | HM3D / MP3D 评测主战场 | `habitat-sim 0.2.4`（基于 Magnum + bullet） |
> | SemExp Gibson | `python vlfm/semexp_env/eval.py` | 对照 SemExp 论文复现 | habitat-sim 0.1.5 旧版 |
> | Spot 真机 | `python vlfm/reality/run_bdsw_objnav_env.py` | Boston Dynamics Spot 落地 | 真机 + bosdyn-client |
>
> 后面以 Habitat 为主线，必要处提及另两套。

## 6.1 仿真本体：habitat-sim + habitat-lab + habitat-baselines

VLFM 站在三层 Habitat 栈上：

```
┌──────────────────────────────────────────────────────────────┐
│  habitat-baselines 0.2.420230405   (vlfm/utils/vlfm_trainer) │  ← evaluator/trainer 循环
├──────────────────────────────────────────────────────────────┤
│  habitat-lab       0.2.420230405                              │  ← ObjectNavTask, sensors, measurements
├──────────────────────────────────────────────────────────────┤
│  habitat-sim       0.2.4 (aihabitat headless_bullet)          │  ← 渲染 + 碰撞 + agent kinematics
└──────────────────────────────────────────────────────────────┘
```

- **habitat-sim**：C++ 渲染 + 物理引擎，吃 `.glb` 场景，吐 RGB / depth / semantic 图片。它在 GPU 上 EGL 离屏渲染，**不开窗**。
- **habitat-lab**：Python 包装 + 任务定义（ObjectNav-v1、PointNav-v1）+ 传感器（GPS / Compass / Heading / ObjectGoalSensor）+ 评估指标（SPL、Success、DistanceToGoal）。
- **habitat-baselines**：训练/评测主循环 + 向量化 env (`VectorEnv`) + obs transforms + PPO/DDPPO 框架。VLFM 把它的 `PPOTrainer._eval_checkpoint` 改写成 `VLFMTrainer`。

依赖在 `pyproject.toml` 里通过 `pip install -e .[habitat]` 装上 habitat-lab/baselines；habitat-sim 通常用 conda 装（`aihabitat`）。

### 6.1.1 数据集

```
data/
├── scene_datasets/
│   └── hm3d/00800-TEEsavR23oF/<scene_id>.glb   ← 3D mesh（800+ 场景）
└── datasets/
    └── objectnav/hm3d/v1/
        ├── train/, val/, val_mini/
        └── val/val.json.gz     ← 含每个 episode 的起点/目标类别
```

`val.json.gz`（验证集）结构：

```json
{
  "category_to_task_category_id": {
    "chair": 0, "bed": 1, "plant": 2,
    "toilet": 3, "tv_monitor": 4, "sofa": 5
  },
  "episodes": [
    {
      "episode_id": "0",
      "scene_id": "hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb",
      "start_position": [7.18678, 2.06447, 4.88622],
      "start_rotation": [0, 0.23418, 0, 0.97219],
      "object_category": "bed",
      "info": { "geodesic_distance": 5.98, "euclidean_distance": 7.86 },
      ...
    }, ...
  ]
}
```

**HM3D ObjectNav 共 6 个目标类**（与 `habitat_policies.py:HM3D_ID_TO_NAME` 对应；注意 yaml 里 `plant` ↔ 代码里 `potted plant`，是一一映射）：

```python
HM3D_ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]
```

MP3D 用 21 类（见 `MP3D_ID_TO_NAME`，含 `table`、`picture`、`cabinet`、`bathtub` 等家具）。

## 6.2 配置入口与传感器声明

VLFM 走 Hydra 配置，主入口是 `config/experiments/vlfm_objectnav_hm3d.yaml`，它 compose 出以下东西：

```yaml
defaults:
  - /habitat_baselines: habitat_baselines_rl_config_base
  - /benchmark/nav/objectnav: objectnav_hm3d        # 任务+数据集+默认相机
  - /habitat/task/lab_sensors:
      - base_explorer       # 第三方 (frontier_exploration 包)
      - compass_sensor
      - gps_sensor
      - heading_sensor
      - frontier_sensor     # 第三方 (frontier_exploration 包)
  - /habitat/task/measurements:
    - frontier_exploration_map
    - traveled_stairs       # 自定义 (vlfm/measurements/traveled_stairs.py)
  - /habitat_baselines/rl/policy: vlfm_policy
```

Compose 完成后，每次 `env.step()` 返回的 `observations: TensorDict` 大致是：

| key | shape | dtype | 来源 |
| --- | --- | --- | --- |
| `rgb` | (1, 480, 640, 3) | uint8 | `HabitatSimRGBSensor` |
| `depth` | (1, 480, 640, 1) | float32 | `HabitatSimDepthSensor`, 归一化到 [0,1] |
| `gps` | (1, 2) | float32 | `GPSSensor`，米 (x, y) |
| `compass` | (1, 1) | float32 | `CompassSensor`，弧度 yaw |
| `heading` | (1, 1) | float32 | `HeadingSensor`，弧度 yaw（与 compass 等价但有偏移） |
| `objectgoal` | (1, 1) | int64 | `ObjectGoalSensor`，0~5 表示目标类别 ID |
| `frontier_sensor` | (1, N, 2) 或 (1, 1, 2) 全零 | float32 | `FrontierSensor`（来自 frontier_exploration 包） |
| `base_explorer` | (1, 1) | uint8 | `BaseExplorer`（这是个**含动作的**特殊 sensor，Oracle FBE 才用） |

`Resize` obs_transformer (`vlfm/obs_transformers/resize.py`) 可以把 RGB / depth 缩到统一大小（默认不缩，HM3D 直接 480×640）。它通过 `apply_obs_transforms_batch(batch, self.obs_transforms)` 在 trainer 里被调用。

> ⚠️ 注意：HM3D 默认 `agent_radius=0.18, height=0.88`，深度传感器位置 `[0, 0.88, 0]`（即头顶 0.88 m），HFOV=79°，min/max_depth=0.5/5.0 m。这些数从 `config/experiments/vlfm_objectnav_hm3d.yaml` 通过 `/benchmark/nav/objectnav: objectnav_hm3d` 继承，可以从 `outputs/<日期>/.hydra/config.yaml` 里查到运行时实际值。

### 6.2.1 自定义传感器：`frontier_sensor` 和 `base_explorer`

它们都来自 [`frontier_exploration`](https://github.com/naokiyokoyama/frontier_exploration) 包（VLFM 作者自己的另一个仓）：

- **`BaseExplorer`**（`frontier_exploration/base_explorer.py`）：
  - **本身是一个 sensor，但实际上输出的是一个"教师动作"**（用于 Oracle Frontier-Based Exploration 基线）。
  - 内部维护 habitat-sim 全局 top-down map 和 fog-of-war，A* 搜出"去最近 frontier 的下一动作"。
  - VLFM **不直接用它的动作**（VLFM 用 `ObstacleMap` 自己生成 frontier、用 ValueMap 选 frontier），但保留 sensor 是为了让 `FrontierSensor` 能访问它。
  - 关键参数（来自实际运行 hydra config）：
    ```
    area_thresh:      3.0      # 米²（注意是 3.0，VLFM 自己的 ObstacleMap 是 1.5）
    forward_step_size: 0.25
    fov:              79
    lin_vel:          0.25
    map_resolution:   256      # 它自己的 top-down 分辨率
    minimize_time:    true
    success_distance: 0.18
    turn_angle:       30.0
    visibility_dist:  4.5
    ```
- **`FrontierSensor`** (`frontier_exploration/frontier_sensor.py`)：
  - 从 BaseExplorer 拿 `frontier_waypoints`（像素坐标），转为 episodic xy。
  - 按 `path_time_cost` 排序后返回 `(N, 2)`；没 frontier 返回 `(1, 2)` 全 0。
  - 这就是 `_observations_cache["frontier_sensor"]` 的来源。

> VLFM 里有 `compute_frontiers` 标志：HM3D 默认 `True`，意味着 **`ObstacleMap.frontiers` 实际上覆盖了 `FrontierSensor` 的输出**——见 `habitat_policies.py:_cache_observations`：
>
> ```python
> if self._compute_frontiers:
>     self._obstacle_map.update_map(depth, ...)
>     frontiers = self._obstacle_map.frontiers
>     self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)
> else:
>     if "frontier_sensor" in observations:
>         frontiers = observations["frontier_sensor"][0].cpu().numpy()
>     else:
>         frontiers = np.array([])
> ```
>
> 也就是说 `frontier_sensor` 是兜底用的；HM3D 评测时实际跑的是 VLFM 自家的 ObstacleMap frontier。

### 6.2.2 自定义 measurement：`traveled_stairs`

`vlfm/measurements/traveled_stairs.py` 注册了一个度量 `TraveledStairs`：

```python
def update_metric(self, *args, **kwargs):
    curr_z = self._sim.get_agent_state().position[1]
    self._history.append(curr_z)
    self._metric = int(np.ptp(self._history) > 0.9)  # 峰峰值 > 0.9m → 走过楼梯
```

它在 episode 结束时被 `episode_stats_logger` 读出来，用于过滤"明明应该停在同一层却跑到楼上"的失败 episode。

### 6.2.3 动作空间

ObjectNav HM3D 默认有 4 个离散动作：

```python
class TorchActionIDs:
    STOP         = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT    = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT   = torch.tensor([[3]], dtype=torch.long)
```

`forward_step_size=0.25 m`，`turn_angle=30°`。`STOP` 必须由策略主动选才能 success（且离目标 < `SUCCESS_DISTANCE=0.1m`，HM3D 默认 0.1，semexp_env 是 0.2）。

## 6.3 顶层调度：`vlfm/run.py` → `VLFMTrainer._eval_checkpoint`

### 6.3.1 启动脚本

```bash
# 终端 1：先启动 4 个 VLM Flask 服务（GroundingDINO/BLIP2-ITM/SAM/YOLOv7）
bash scripts/launch_vlm_servers_jy.sh 0   # 用 GPU 0 部署

# 终端 2：跑评测
python -m vlfm.run \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=data/dummy_policy.pth \
  habitat_baselines.rl.policy.name=HabitatITMPolicyV2 \
  habitat.task.lab_sensors.base_explorer.turn_angle=30 \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.split=val \
  habitat_baselines.eval.video_option='["disk"]'
```

`vlfm/run.py:42-55`：

```python
@hydra.main(config_path="../config", config_name="experiments/vlfm_objectnav_hm3d")
def main(cfg):
    cfg = patch_config(cfg)
    with read_write(cfg):
        try:
            cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
        except KeyError:
            pass
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")
```

- 把 semantic_sensor 关掉（ObjectNav 是零知识场景，不能作弊）。
- `execute_exp` 在 habitat-baselines 里查 `trainer_name="vlfm"` → 创建 `VLFMTrainer` 实例 → 调 `_eval_checkpoint`。

### 6.3.2 VLFMTrainer 评测循环

`vlfm/utils/vlfm_trainer.py:_eval_checkpoint`，每帧流程：

```
env.reset()                    # 第一次或一集结束时
  ↓ observations: list[Dict]
batch_obs(observations) → batch: TensorDict on CUDA
apply_obs_transforms_batch     # Resize 等
  ↓
self._agent.actor_critic.act(batch, hidden, prev_actions, masks)
  ↓ PolicyActionData(actions, rnn_hidden_states, policy_info)
step_data = a.item() for a in action_data.env_actions.cpu()
envs.step(step_data)
  ↓ list of (obs, reward, done, info)
hab_vis.collect_data(batch, infos, policy_info)   # 视频帧
if done: generate_video(...)
```

要点：
- `self._agent` 在 habitat-baselines 里被 `_create_agent` 创建。VLFM 的 `actor_critic` 是 `HabitatITMPolicyV2`（继承 `BaseObjectNavPolicy`）。
- **`actor_critic.act` 的就是我们在 04 章讲过的那个 `act`**——只是被 habitat-baselines 包了一层 `PolicyActionData`。
- `envs` 是 `VectorEnv`，**多环境并行**，但 VLFM 配 `num_environments=1`（因为各种地图状态都是单 env 单例）。
- `done=True` 时自动重置；`hab_vis.flush_frames` 把累积的可视化帧打包成 mp4 落到 `video_dir/`。
- TensorBoard scalar 写到 `tensorboard_dir`。

## 6.4 一帧 RGBD 是怎么从 sim 跑到 ObstacleMap/ValueMap 的？

这是你最关心的"如何环境得到 rgbd 给代码"。下面按时序展开。

### 6.4.1 sim → habitat-lab → habitat-baselines

```
habitat-sim RenderTarget (CUDA 上)
   │
   ▼ 触发 sensor read
HabitatSimRGBSensor.get_observation()  → np.uint8 (480, 640, 3)
HabitatSimDepthSensor.get_observation()
   │ depth 在 sim 内部除以 max_depth(5.0) 归一化到 [0,1]
   │ (来自 sim_sensors.depth_sensor.normalize_depth=true)
   ▼
ObjectNavTask.step(action) 收集所有 sensors → observations: dict
   │
   ▼ env.step()
VectorEnv: list[dict] 一个元素一个并行 env
   │
   ▼ trainer 里 batch_obs
batch: TensorDict (cuda)
{
  "rgb":    Tensor (1, 480, 640, 3) uint8 / cuda
  "depth":  Tensor (1, 480, 640, 1) float32 / cuda
  ...
}
   │
   ▼ apply_obs_transforms_batch (Resize 等)
batch unchanged (HM3D 默认不 resize)
   │
   ▼ self._agent.actor_critic.act(batch, ...)
```

### 6.4.2 进入 HabitatITMPolicyV2 后

`HabitatMixin._cache_observations`（`habitat_policies.py:173-237`）把 batch 拆开存进 `_observations_cache`：

```python
rgb         = observations["rgb"][0].cpu().numpy()         # (480, 640, 3) uint8
depth       = observations["depth"][0].cpu().numpy()       # (480, 640, 1) float32
x, y        = observations["gps"][0].cpu().numpy()         # (2,) float32, 米
camera_yaw  = observations["compass"][0].cpu().item()      # float, 弧度

depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                                                            # → (480, 640) 修复小空洞

camera_position = np.array([x, -y, self._camera_height])    # 注意 y 翻号！
                                                            # Habitat GPS 把西定义为负 y
                                                            # VLFM 内部把它翻成 episodic frame
robot_xy        = camera_position[:2]
tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
                                                            # (4, 4) 同构变换
```

紧接着 ObstacleMap 拿 `depth` 第一时间更新：

```python
if self._compute_frontiers:
    self._obstacle_map.update_map(
        depth, tf_camera_to_episodic,
        self._min_depth, self._max_depth,
        self._fx, self._fy, self._camera_fov,
    )
    frontiers = self._obstacle_map.frontiers
```

然后所有需要 RGBD 的下游都打包成 list 存进 cache：

```python
self._observations_cache = {
    "frontier_sensor": frontiers,
    "nav_depth":       observations["depth"],              # 给 PointNav 用 (raw tensor)
    "robot_xy":        robot_xy,
    "robot_heading":   camera_yaw,
    "object_map_rgbd": [(rgb, depth, tf, 0.5, 5.0, fx, fy)],   # 给 _update_object_map
    "value_map_rgbd":  [(rgb, depth, tf, 0.5, 5.0, fov)],      # 给 _update_value_map
    "habitat_start_yaw": observations["heading"][0].item(),
}
```

> 关键：**RGB / depth 此时已经从 GPU Tensor 转到 CPU numpy**。这是因为后续要送给 Flask 服务（HTTP/JSON 传输，必须是 numpy）。

### 6.4.3 一帧 rgbd 怎么走到 BLIP2 / YOLO

ValueMap 路径（`itm_policy.py:_update_value_map`）：

```python
all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
cosines = [
    [self._itm.cosine(rgb, prompt.replace("target_object", self._target_object))
     for prompt in self._text_prompt.split("|")]
    for rgb in all_rgb
]
for cosine, (rgb, depth, tf, mind, maxd, fov) in zip(cosines, ...):
    self._value_map.update_map(np.array(cosine), depth, tf, mind, maxd, fov)
```

`self._itm.cosine(rgb, prompt)` 内部走 RPC：

```
BLIP2ITMClient.cosine(rgb, "Seems like there is a bed ahead.")
   │
   ▼ vlfm/vlm/server_wrapper.py:send_request
JPEG encode rgb (quality=90) → base64 string
payload = {"image": "<base64>", "txt": "Seems like there is a bed ahead."}
POST http://localhost:12182/blip2itm   timeout=1s, retry up to 20s
   │
   ▼ Flask server (vlfm/vlm/blip2itm.py:BLIP2ITMServer)
str_to_image → np.ndarray → BLIP2 model → cosine ∈ [0, 0.6]
return {"response": cosine}
   │
   ▼
float(response["response"])
```

ObjectMap 路径（`base_objectnav_policy.py:_update_object_map`）：

```python
for (rgb, depth, tf, mind, maxd, fx, fy) in object_map_rgbd:
    detections = self._get_object_detections(rgb)       # → YOLO 或 GroundingDINO
    for idx in range(len(detections.logits)):
        bbox = detections.boxes[idx] * np.array([W, H, W, H])
        object_mask = self._mobile_sam.segment_bbox(rgb, bbox.tolist())  # → MobileSAM
        self._object_map.update_map(target, depth, object_mask, tf, mind, maxd, fx, fy)
```

YOLO/Grounding/SAM 各自的 RPC 也都是 `image_to_str(JPEG, q=90)` + 字段 payload。

## 6.5 通信细节：`server_wrapper.py` 的进程间约定

`vlfm/vlm/server_wrapper.py` 是个非常简单的 Flask + lockfile 设计：

### 6.5.1 服务端

```python
def host_model(model, name, port):
    app = Flask(__name__)
    @app.route(f"/{name}", methods=["POST"])
    def process_request():
        payload = request.json
        return jsonify(model.process_payload(payload))
    app.run(host="localhost", port=port)
```

每个 VLM 模型有自己的 endpoint：
- `localhost:12181/gdino` → GroundingDINO
- `localhost:12182/blip2itm` → BLIP2-ITM (`cosine`)
- `localhost:12183/mobile_sam` → MobileSAM
- `localhost:12184/yolov7` → YOLOv7
- `localhost:12185/blip2` → BLIP2 VQA（可选，`use_vqa=True` 才用）

由 `scripts/upstream/launch_vlm_servers.sh` 用 tmux 拉起 4 panes，模型加载好后驻留 GPU 显存（每次 RPC 不重新加载）。

### 6.5.2 客户端（VLFM 这边）

`send_request(url, **kwargs)`：

1. 把所有 `np.ndarray` 用 `cv2.imencode('.jpg', ..., quality=90)` → base64 → 字符串。
2. POST `{"image": "<b64>", "txt": "..."}`，timeout=1s。
3. 失败重试 20s，超过 20s 抛异常。
4. **lockfile 互斥**：在 `lockfiles/` 目录创建 `http_localhost_12182_blip2itm.lock` 文件，避免多个 VLFM 客户端同时打 server 把 GPU 撑爆。文件超过 120s 自动清理。

### 6.5.3 SAM 的 mask 怎么传回来

mask 是 bool 数组，先 `bool_arr_to_str` 用 base64 压成字符串传过来；客户端 `str_to_bool_arr(s, shape=image.shape[:2])` 解回 `(H, W) bool`。

### 6.5.4 通信开销

实测（GPU 0 上 4 个 VLM + GPU 7 跑 sim）：
- BLIP2-ITM 一次 cosine：~50 ms（含 JPEG 编码、HTTP、模型推理）
- YOLOv7 一次：~30 ms
- GroundingDINO 一次：~150 ms
- SAM 一次：~80 ms（每个 bbox）
- 总每步约 200~400 ms。可视化 + 视频 IO 还会再翻倍。

如果想加速，可以把 sim 和 VLM 放同一进程（去掉 HTTP）；但 Flask 设计让 4 个模型可以独立挑卡，跨机器也能跑。

## 6.6 Habitat 默认相机参数表（HM3D）

来自 `outputs/<日期>/.hydra/config.yaml` 实测：

| 项目 | 值 | 说明 |
| --- | --- | --- |
| `rgb_sensor.type` | HabitatSimRGBSensor | – |
| `rgb_sensor.height/width` | 480 / 640 | px |
| `rgb_sensor.hfov` | 79 | 度 |
| `rgb_sensor.position` | [0, 0.88, 0] | 米（agent 中心 +0.88m 高度） |
| `rgb_sensor.sensor_subtype` | PINHOLE | – |
| `depth_sensor.height/width` | 480 / 640 | 同 RGB |
| `depth_sensor.min_depth / max_depth` | 0.5 / 5.0 | 米 |
| `depth_sensor.normalize_depth` | true | sim 内部 `depth ← depth / max_depth` |
| `agent.height / radius` | 0.88 / 0.18 | 米 |
| `forward_step_size` | 0.25 | 米 |
| `turn_angle / tilt_angle` | 30 / 30 | 度 |
| `simulator.action_space_config` | v1 | 4 离散动作 |
| `allow_sliding` | False | 撞墙不会滑 |

由 fx/fy 公式：`fx = fy = W / (2·tan(hfov/2)) = 640 / (2·tan(79°/2)) ≈ 390.6 px`。

## 6.7 Reality (Spot) 环境 与 Habitat 的接口对齐

`vlfm/reality/objectnav_env.py:ObjectNavEnv` 是个标准 Gym 风格 env，但 obs 直接组装到与 Habitat 缓存等价的字典里（**绕开了 habitat-baselines 那一套 batch/transform**）：

```python
def _get_obs(self):
    robot_xy, robot_heading = self._get_gps(), self._get_compass()
    nav_depth, obstacle_map_depths, value_map_rgbd, object_map_rgbd = self._get_camera_obs()
    return {
        "nav_depth":          nav_depth,
        "robot_xy":           robot_xy,
        "robot_heading":      robot_heading,
        "objectgoal":         self.target_object,
        "obstacle_map_depths": obstacle_map_depths,
        "value_map_rgbd":      value_map_rgbd,
        "object_map_rgbd":     object_map_rgbd,
    }
```

注意三处差异：

1. **Spot 同时有 5 个深度相机**（前左/前右/左/右/后）和 1 个手部 RGB；`object_map_rgbd` 只用手部 RGB，`obstacle_map_depths` 用 5 个深度合成。
2. **手部 RGB 没有伴随的深度**（fisheye RGB），所以 `value_map_rgbd` 里 depth 用 **ZoeDepth 单目深度估计**填充（`reality_policies.py:_infer_depth`）；object_map 里则用 `np.ones(...)` 占位，由 `_update_object_map` 检测到目标后再调 `_infer_depth` 替换。
3. **动作是连续的** `(angular_vel, linear_vel)`，而不是离散 STOP/FWD/TL/TR；停止条件改成 `arm_yaw==-1`。

但**所有 `BaseObjectNavPolicy` 下游代码完全不变**——因为 `_observations_cache` 的 schema 一致。这就是 VLFM 论文说"zero-shot transfer to real robot"的工程基础：mock 同样的字典。

## 6.8 SemExp Gibson 环境（Legacy）

`vlfm/semexp_env/eval.py` 走的是另一套老的 habitat 0.1.5 + SemExp 仓库的 `make_vec_envs`：

```python
envs = make_vec_envs(args)            # SemExp 的封装
obs, infos = envs.reset()
for step in range(args.max_episode_length):
    obs_dict = merge_obs_infos(obs, infos)    # 把 SemExp 的 [B, 4, H, W] 拆成 rgb/depth
    action, policy_infos = policy.act(obs_dict, None, None, masks)
    obs, rew, done, infos = envs.step(action)
```

`merge_obs_infos`(`semexp_env/eval.py:128-148`) 是关键适配层：

```python
rgb   = obs[:, :3, ...].permute(0, 2, 3, 1)   # [1, 3, H, W] → [1, H, W, 3]
depth = obs[:, 3:4, ...].permute(0, 2, 3, 1)
obs_dict = {
    "rgb": rgb, "depth": depth,
    "objectgoal":  info_dict["goal_name"].replace("-", " "),
    "gps":         tensor_from_numpy(obs, info_dict["gps"]).unsqueeze(0),
    "compass":     tensor_from_numpy(obs, info_dict["compass"]).unsqueeze(0),
    "heading":     tensor_from_numpy(obs, info_dict["heading"]).unsqueeze(0),
}
```

即把 SemExp 的 `(1, C, H, W)` channel-first tensor 还原成 Habitat 的 `(1, H, W, C)` channel-last，让同一个策略复用。

## 6.9 视频与日志输出

| 路径 | 内容 |
| --- | --- |
| `outputs/<日期>/<时间>/.hydra/config.yaml` | hydra 序列化的完整运行配置（可查 sensor / fov / 阈值） |
| `outputs/<日期>/<时间>/<其他>` | habitat-baselines 训练/评测日志 |
| `tb/<run_id>/` | TensorBoard 标量（success rate、SPL 等） |
| `video_dir/<run_id>/epid=XXX-scid=...-succ=1-spl=0.85-...mp4` | 每集一个 mp4 视频，左 RGB+depth、右 top-down + obstacle_map + value_map |
| `vis/<日期-时间>/` | Reality 模式 Spot 每步保存 4 张图（annotated_rgb/depth、obstacle_map、value_map）+ 合成 jpg |
| `value_map_recordings/` | 当 `RECORD_VALUE_MAP=1` 时记录每帧 depth + tf + values 的 JSON+PNG（用于离线重放调参） |
| `lockfiles/` | VLM RPC 客户端互斥锁，运行结束应为空 |

## 6.10 完整生命周期（一集 episode）

```
[启动期]
  bash scripts/launch_vlm_servers_jy.sh 0        # 后台拉起 GroundingDINO/BLIP2-ITM/SAM/YOLOv7
  python -m vlfm.run ...                          # hydra 解析配置
       ↓
  execute_exp(cfg, "eval")
       ↓
  VLFMTrainer._eval_checkpoint
       ↓
  envs.reset() ─► habitat-sim 加载 hm3d/<scene_id>.glb → 把 agent 放到 start_position
                  ObjectNavTask.set_objectgoal(category_id)
                  各 sensor.reset()
       ↓
  observations[0] = {rgb, depth, gps, compass, heading, objectgoal, frontier_sensor, base_explorer}

[Step Loop]  while not done:
  batch = batch_obs(observations)
  batch = apply_obs_transforms_batch(batch, [Resize])
       ↓
  action_data = HabitatITMPolicyV2.act(batch, hidden, prev_action, mask)
       │
       ├── HabitatMixin.act: objectgoal id → "bed"
       ├── ITMPolicyV2.act:
       │     ├── _pre_step: ObstacleMap.update_map, cache 一帧 obs
       │     ├── _update_value_map: 走 4×RPC (BLIP2-ITM) → ValueMap 累积
       │     └── BaseObjectNavPolicy.act:
       │           ├── _update_object_map: YOLO 或 Grounding → SAM → ObjectPointCloudMap
       │           ├── goal = _get_target_object_location() or None
       │           ├── if init: TURN_LEFT
       │           │  elif goal: pointnav(goal, stop=True)
       │           │  else: explore() → sort frontiers → pointnav(best)
       │           └── _get_policy_info → 给可视化用
       │
       └── 返回 PolicyActionData(actions=Tensor[[1]], ...)

  envs.step(action) ─► habitat-sim 应用 MOVE_FORWARD 0.25m → 渲染新 RGB-D
       ↓
  observations, reward, done, info = ...
  if done:
      generate_video(rgb_frames, episode_id, ..., spl, succ)
      log_episode_stats(episode_id, scene_id, info)        # 写 jsonl
      envs.reset()   # 进入下一集
```

## 6.11 你最容易踩的坑

| 现象 | 原因 |
| --- | --- |
| `Connection refused` 或 cosine 一直超时 | `launch_vlm_servers_jy.sh` 没跑 / 模型还在加载（要 ~60-90s） |
| 重放 `replay_from_dir` 出现 0 余弦 | 没设 `RECORD_VALUE_MAP=1` 当时；或 `kwargs.json` 不存在 |
| 永远走 `explore` 模式 | YOLO 阈值 0.8 太高（COCO 类）或 GroundingDINO 抓不到（caption 没拼对）|
| Habitat 直接 OOM | `num_environments` >1 但 GPU 显存吃紧；ValueMap+ObstacleMap 是单 env 单例，并行会跑飞 |
| FrontierSensor 一直返回 `(1, 2)` 全 0 | `BaseExplorer` 还没把 fog-of-war 揭开足够区域；与 VLFM 自己的 ObstacleMap.frontiers 是两套，前者只是兜底 |
| 录的视频里 value_map 一片白 | `visualize=False`（`len(eval.video_option)==0`）；改 `eval.video_option='["disk"]'` |
| `cannot connect to GPU 7 from BLIP2` | `launch_vlm_servers_jy.sh` 默认绑 GPU 0；想换卡传参 `bash scripts/launch_vlm_servers_jy.sh 7` |

## 6.12 一句话总结

VLFM 的 Habitat 仿真本体是 **habitat-sim 0.2.4 + habitat-lab/baselines 0.2.420230405**；环境通过 **`HabitatSimRGBSensor / HabitatSimDepthSensor`** 在 GPU 上离屏渲染出 `(480, 640, 3) uint8` 和 `(480, 640, 1) float32 ∈ [0,1]` 的 RGB-D，配合 `gps/compass/heading/objectgoal/frontier_sensor` 一起打包成 `TensorDict`，由 **`VLFMTrainer._eval_checkpoint`** 在 PyTorch CUDA 上 `batch_obs` 后送入 `HabitatITMPolicyV2.act`；策略内部把 RGB/depth 拷到 CPU numpy，通过 **`server_wrapper.send_request`** 经 4 个本地 Flask 端口（12181/12182/12183/12184）做 RPC 调用 GroundingDINO/BLIP2-ITM/MobileSAM/YOLOv7；产出再回填到 `ObjectPointCloudMap / ValueMap / ObstacleMap`，最终 `(ρ, θ)` 喂 PointNav ResNet-LSTM 得到离散动作回到 `envs.step()`。整个回路约 200-400 ms/帧。
