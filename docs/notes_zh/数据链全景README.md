# VLFM 数据链全景（habitat-sim ⇄ 策略 ⇄ VLM 服务）

> 本文按你给的视角顺一遍 VLFM 一个 step 内的数据链：
>
> habitat-sim 出 RGBD/位姿 → `HabitatMixin._cache_observations` 缓存 → BLIP2-ITM 锥形投影构 `ValueMap` → 深度投点云构 `ObstacleMap` + frontier → YOLO/G-DINO+MobileSAM 抠目标 mask → 反投影 DBSCAN 形成目标点云簇 → 决策（initialize / explore / navigate）→ PointNav 出 4-类离散动作 → 回灌 habitat-sim。
>
> 默认配置走的策略类是 **`HabitatITMPolicyV2`**（`vlfm/config/experiments/vlfm_objectnav_hm3d.yaml` 中 `rl.policy.name`）。
>
> 类继承关系（自上而下）：
>
> ```text
> habitat_baselines.rl.ppo.Policy
> └── BasePolicy                       vlfm/policy/base_policy.py
>     └── BaseObjectNavPolicy          vlfm/policy/base_objectnav_policy.py
>         └── BaseITMPolicy            vlfm/policy/itm_policy.py
>             └── ITMPolicyV2          vlfm/policy/itm_policy.py
>                 └── HabitatITMPolicyV2 = HabitatMixin + ITMPolicyV2
>                                       vlfm/policy/habitat_policies.py
> ```

---

## 0. 一图速记

```text
                                                        ┌───────────────────┐
                                                        │  habitat-sim      │
                                                        │  (HabitatSim)     │
                                                        └─────────┬─────────┘
                                                                  │   per-step
                                                                  │   observations: TensorDict
                                                                  ▼
                                            ┌──────────────────────────────────────────┐
                                            │ habitat_baselines rollout / trainer      │
                                            │ + Resize ObservationTransformer (224x224)│
                                            └─────────────────────┬────────────────────┘
                                                                  │
                                                  HabitatITMPolicyV2.act(observations,…)
                                                                  │
                                                                  ▼
            ┌─────────────────────────────────────────────────────────────────────────────────────┐
            │                          HabitatMixin._cache_observations                            │
            │  rgb (H,W,3)  depth (H,W) ∈[0,1]  gps (x,y)  compass yaw  heading start_yaw          │
            │  frontier_sensor (N,2) (来自 frontier_exploration sensor，或本地 ObstacleMap)        │
            │  → tf_camera_to_episodic (4×4)                                                      │
            └────────────┬───────────────────────────┬──────────────────────────────┬─────────────┘
                         │ value_map_rgbd            │ object_map_rgbd              │ obstacle map（同源 depth）
                         ▼                           ▼                              ▼
              BLIP2-ITM @ port 12182        YOLOv7/G-DINO @ 12184/12181     ObstacleMap.update_map
              余弦 → cosine ∈ [0,1]         + MobileSAM @ 12183             → 障碍/可航行/已探索/frontier
                         │                           │                              │
                         ▼                           ▼                              │
              ValueMap.update_map           ObjectPointCloudMap.update_map          │
              （锥形 mask × cosine 融合）    （depth+mask→点云→DBSCAN→episodic）     │
                         │                           │                              │
                         └────────────┬──────────────┴────────────┬─────────────────┘
                                      │                           │
                              has_object(target)?         frontier_sensor
                                      │                           │
                                      ▼                           ▼
                                 navigate 模式                explore 模式（按 ValueMap 排 frontier）
                                      └───────────┬───────────────┘
                                                  ▼
                          rho, theta + 224×224 depth → WrappedPointNavResNetPolicy
                                                  │
                                                  ▼
                                       action ∈ {0:STOP,1:FWD,2:L,3:R}
                                                  │
                                                  ▼
                                          回灌 habitat-sim 执行一步
```

---

## 1. habitat-sim 到底提供了什么

入口在 `vlfm/run.py`：

- `frontier_exploration` / `vlfm.measurements.traveled_stairs` / `vlfm.obs_transformers.resize` / `vlfm.policy.action_replay_policy` / `vlfm.policy.habitat_policies` / `vlfm.utils.vlfm_trainer` 这一坨 `# noqa` 的 import，都是为了在 Hydra 启动时把这些类**注册**进 `baseline_registry`，从而被配置文件按名字找到。
- `@hydra.main(... config_name="experiments/vlfm_objectnav_hm3d")` → 进 `habitat_baselines.run.execute_exp` → 真正的训练/评估循环。

实际生效的传感器/动作/数据集都在配置里：

```startLine:endLine:filepath
config/experiments/vlfm_objectnav_hm3d.yaml
```

关键 default：
- `/benchmark/nav/objectnav: objectnav_hm3d` — 这条把 ObjectNav-HM3D 数据集和 4 个离散动作 `STOP/MOVE_FORWARD/TURN_LEFT/TURN_RIGHT` 带进来。
- `/habitat/task/lab_sensors:` 注册了：
  - `base_explorer`：Oracle 探索器，能直接吐 frontier 动作（只给 oracle 策略用）。
  - `compass_sensor`：当前 yaw（弧度，相对 episode 起始）。
  - `gps_sensor`：当前 (x, y)（米，episode 局部系，**y 是西负**）。
  - `heading_sensor`：episode 开始时的 heading（用来对齐可视化）。
  - `frontier_sensor`：直接从 habitat 自带的 top-down map + fog-of-war 算 frontier 候选点（episodic xy）。
- `/habitat/task/measurements:` 加了 `frontier_exploration_map` 和 `traveled_stairs`（评估指标用）。
- `/habitat_baselines/rl/policy: vlfm_policy` 把 `VLFMPolicyConfig` 注入到 `habitat_baselines.rl.policy`。

每一步 habitat 给到策略的 `observations: TensorDict` 里至少有：

| key | 形状/类型 | 含义 |
| --- | --- | --- |
| `rgb` | `(1, H, W, 3) uint8` | RGB 图（经过 `Resize` obs-transformer 一般是 (1,224,224,3)） |
| `depth` | `(1, H, W, 1) float32 ∈ [0,1]` | 归一化深度，需要乘 `(max_depth-min_depth)+min_depth` 还原成米 |
| `gps` | `(1, 2) float32` | episode 局部 (x, y)，米 |
| `compass` | `(1, 1) float32` | 当前朝向，弧度 |
| `heading` | `(1, 1) float32` | episode 启动时的朝向，弧度 |
| `objectgoal` | `(1, 1) int64` | 目标类别 id（HM3D: 0-5, MP3D: 0-20） |
| `frontier_sensor` | `(1, N, 2) float32` | 来自 frontier_exploration sensor 的 frontier 候选 episodic xy（米） |

> 这里要小心两件事：
> 1. `Resize` obs-transformer 默认把 rgb/depth/semantic 都缩到 (224,224)；rgb 实际仍是 480×640 还是 224×224，看你拿到 obs_dict 的时机（trainer 会在送进策略前先过 transformer）。`HabitatMixin._cache_observations` 拿到的是 transformer 之后的。
> 2. `gps` 的 y 是西负，所以 `HabitatMixin._cache_observations` 里写了 `camera_position = np.array([x, -y, self._camera_height])`，把 y 翻号到内部一致的 episodic 系。

策略类配置（`HabitatMixin.from_config`）会从 sim 配置里读相机参数（高度、min/max 深度、HFOV、宽度）算焦距：

```startLine:endLine:filepath
vlfm/policy/habitat_policies.py
camera_fov_rad = np.deg2rad(camera_fov)
self._camera_fov = camera_fov_rad
self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))
```

例：HM3D 默认 `HFOV=79°`, `width=640`，得 `fx = 640 / (2 * tan(79°/2)) ≈ 393.3`。如果 Resize 把 width 改成 224，HabitatMixin 用的还是 sim 给的原始 width，所以 fx/fy 对应的是 sim 原图分辨率——这跟 `_extract_object_cloud` 里 `cloud = get_point_cloud(valid_depth, final_mask, fx, fy)` 是要 `depth.shape[1]` 一致的，所以注意 `depth` 缓存的是 transformer 之后的归一化深度（默认 224×224），但 fx/fy 是按 sim 原宽算的——VLFM 默认 HM3D 跑下来 depth.shape 与 width 是一致的（都是 224 或都是 480/640），别人改配置时要核对。

---

## 2. `_cache_observations`：把一帧拍平成"内部统一格式"

> 文件：`vlfm/policy/habitat_policies.py`，类 `HabitatMixin`。
>
> 角色：所有策略子类（`HabitatITMPolicyV2` 等）通过 mixin 拿到一个**只在第一次调用时填充**的 `_observations_cache`，后续 `act()` 内任何模块都从这里取，所以一帧只解一次。

签名：

```python
def _cache_observations(
    self: Union["HabitatMixin", BaseObjectNavPolicy],
    observations: TensorDict,
) -> None
```

副作用（写入 `self._observations_cache`，类型 `Dict[str, Any]`）：

| key | 形状/类型 | 怎么来 | 给谁用 |
| --- | --- | --- | --- |
| `nav_depth` | `Tensor (1, H, W, 1)` | 直接拿 `observations["depth"]` | `_pointnav` 喂给 PointNav 网络 |
| `robot_xy` | `np.ndarray (2,)` | `camera_position[:2]`（y 已翻号） | 各种地图更新、pointnav 起点 |
| `robot_heading` | `float` | `observations["compass"]` | tf 矩阵、地图更新 |
| `frontier_sensor` | `np.ndarray (N,2) 或 (0,)` | 优先 `observations["frontier_sensor"]`；若 `_compute_frontiers=True` 改用本地 `ObstacleMap.update_map` 后的 `.frontiers` | ITMPolicy 选择探索目标 |
| `object_map_rgbd` | `list[tuple]` 长度 1 | `(rgb, depth, tf, min_depth, max_depth, fx, fy)` | YOLO/G-DINO + SAM + 物体点云 |
| `value_map_rgbd` | `list[tuple]` 长度 1 | `(rgb, depth, tf, min_depth, max_depth, fov)` | BLIP2-ITM + ValueMap |
| `habitat_start_yaw` | `float` | `observations["heading"]` | 可视化对齐 |

里面有两步关键变换：

1. **深度滤波**：`depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)`（来自 `depth_camera_filtering`，去掉 0/NaN 等无效像素）。
2. **位姿矩阵**：`tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)`，得到 4×4：

```python
np.array([
    [cos(yaw), -sin(yaw), 0, x],
    [sin(yaw),  cos(yaw), 0, y],
    [0,         0,        1, z=camera_height],
    [0,         0,        0, 1],
])
```

举个具体值（HM3D 第一步，相机高 0.88m，gps=(0,0)，compass=0）：

```python
tf_camera_to_episodic ≈
[[1, 0, 0, 0.0],
 [0, 1, 0, 0.0],
 [0, 0, 1, 0.88],
 [0, 0, 0, 1.0]]
```

如果配置 `compute_frontiers=True`（默认），还会在缓存里**顺手更新一次 ObstacleMap**（见 §4），并把 `obstacle_map.frontiers` 灌进 `frontier_sensor`。这就是为什么哪怕你不开 habitat 的 `frontier_sensor`，策略也照样有 frontier 可用。

---

## 3. ValueMap：BLIP2-ITM 的"锥形价值地图"

### 3.1 入口

`ITMPolicyV2.act` 里第一件事就是 `self._update_value_map()`（注意 V2 是必跑，V1 只有可视化时才跑）：

```startLine:endLine:filepath
vlfm/policy/itm_policy.py
def _update_value_map(self) -> None:
    all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
    cosines = [
        [
            self._itm.cosine(
                rgb,
                p.replace("target_object", self._target_object.replace("|", "/")),
            )
            for p in self._text_prompt.split(PROMPT_SEPARATOR)
        ]
        for rgb in all_rgb
    ]
    for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
        cosines, self._observations_cache["value_map_rgbd"]
    ):
        self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)
```

`self._text_prompt` 默认是 `"Seems like there is a target_object ahead."`（V3 还会拼第二个 prompt 做 exploration_value，所以 `value_channels` 可能为 1 或 2）。

### 3.2 BLIP2ITMClient.cosine（VLM RPC）

文件 `vlfm/vlm/blip2itm.py`：

```python
class BLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])
```

- 入参：`image` 一般是 `(H,W,3) uint8`，`txt` 是 prompt 字符串。
- 副作用：发 HTTP POST 到 12182，服务端 `BLIP2ITM.cosine` 用 LAVIS 的 `blip2_image_text_matching` 模型计算余弦匹配。
- 出参：一个 `float`，**论文里说是 (0,1) 区间的 score**（实际上是 `match_head="itc"` softmax 后的标量）。
- 举例：`cosine(rgb_chair_in_view, "Seems like there is a chair ahead.") ≈ 0.71`。

### 3.3 ValueMap.update_map

签名（`vlfm/mapping/value_map.py`）：

```python
def update_map(
    self,
    values: np.ndarray,          # (value_channels,) float，本步的 cosine
    depth: np.ndarray,           # (H, W) ∈ [0,1]
    tf_camera_to_episodic: np.ndarray,  # (4,4)
    min_depth: float, max_depth: float, # 米
    fov: float,                  # 弧度（水平 FOV）
) -> None
```

内部三步走：

1. `_process_local_data(depth, fov, min_depth, max_depth)` → 返回一个局部"锥形可见区"mask（带置信度，FOV 中心 1，外缘 `_min_confidence=0.25`），并用深度切掉**被遮挡之后**的部分。具体做法：
   - `depth_row = np.max(depth, axis=0) * (max-min) + min` 把每一列取最大深度作为这一方向的可达距离（米）。
   - 把 `depth_row` 沿 `-fov/2…fov/2` 的角度投到 xy 像素坐标，画一条 contour，把"被遮挡区"挖掉（`cv2.drawContours(..., 0, -1)`）。
   - 锥形 mask 用 `_get_confidence_mask` 生成，置信度 ∝ `cos(angle)^2`，存到 `_confidence_masks` 缓存。
   - 形状：`(2*size+1, 2*size+1) float32`，size = `int(max_depth * pixels_per_meter)`（默认 `pixels_per_meter=20`，`max_depth=5m` ⇒ ~201×201）。
2. `_localize_new_data` 把局部 mask 按 `yaw = extract_yaw(tf)` 旋转，再 `place_img_in_img` 贴到全局 1000×1000 大图（米→像素 `_xy_to_px`）。
3. `_fuse_new_data(curr_map, values)` 做**带置信度的多通道融合**：
   - 默认 `use_max_confidence=False`：
     ```
     weight_1 = self._map / (self._map + new_map)
     weight_2 = new_map  / (self._map + new_map)
     self._value_map = self._value_map * weight_1 + values * weight_2   # 每通道独立
     self._map       = self._map       * weight_1 + new_map * weight_2  # 置信度也加权
     ```
   - `use_max_confidence=True`：哪个置信度高就直接覆盖。
   - 还有 `_decision_threshold=0.35`：当新置信度低于 0.35 且也比已有低时直接归零，避免噪声。
   - 如果绑了 `ObstacleMap`（`sync_explored_areas=True`），还会把未探索区域的 value 强制清零。

副作用：写入 `self._value_map: (size, size, value_channels) float32` 和 `self._map: (size, size) float32`（置信度）。

### 3.4 ValueMap.sort_waypoints（**explore 用**）

```python
def sort_waypoints(
    self, waypoints: np.ndarray, radius: float, reduce_fn: Optional[Callable] = None
) -> Tuple[np.ndarray, List[float]]
```

- 入：N 个 frontier 候选 `(N,2)`，搜索半径（米），多通道 reduce 函数。
- 算每个 frontier 点对应像素 ±radius 内的 `pixel_value_within_radius` 中位数，作为该 frontier 的"价值"。
- 出：`(sorted_frontiers (N,2), sorted_values list[float])`，**按 value 降序**。

举例：`sort_waypoints([[2,1],[-1,3]], 0.5)` → `(array([[2,1],[-1,3]]), [0.71, 0.42])`。

### 3.5 总结一句话

ValueMap 本质上是 **"BLIP2 余弦评分在 RGBD 锥形可见区上的置信度加权积分图"**：每一步把当前 cosine 当成该锥内像素的目标价值，按置信度（中心高、边缘低、遮挡为零）和历史值加权融合。

---

## 4. ObstacleMap & frontier：基于深度的可达性 + 边界探测

文件 `vlfm/mapping/obstacle_map.py`。

### 4.1 数据载体

```python
ObstacleMap(min_height, max_height, agent_radius, area_thresh=3.0, hole_area_thresh=100000,
            size=1000, pixels_per_meter=20)
```

- `self._map: (1000,1000) bool` —— 障碍物 topdown，True 表示障碍。
- `self._navigable_map: (1000,1000) bool` —— 障碍 dilate（用 `agent_radius` 圆核）的补集，True 表示可走。
- `self.explored_area: (1000,1000) bool` —— 已探索区，True 表示见过。
- `self.frontiers: (N, 2) float32` —— 边界候选 episodic 米坐标。

### 4.2 update_map

```python
def update_map(
    self,
    depth, tf_camera_to_episodic,
    min_depth, max_depth, fx, fy,
    topdown_fov,                # 来自 self._camera_fov（弧度）
    explore=True, update_obstacles=True,
) -> None
```

流程：

1. **填洞**：`filled_depth = fill_small_holes(depth, hole_area_thresh)` 把小于阈值的 0 像素洞填上。
2. **米化**：`scaled_depth = filled_depth * (max-min) + min`。
3. **mask**：`mask = scaled_depth < max_depth`，太远的不要（防止天花板/远墙噪声）。
4. **3D 点云（相机系）**：`get_point_cloud(scaled_depth, mask, fx, fy)` →
   ```python
   v, u = np.where(mask)
   z = depth[v,u]
   x = (u - W/2) * z / fx
   y = (v - H/2) * z / fy
   cloud = np.stack((z, -x, -y), axis=-1)  # (M, 3)，相机系 x 朝前
   ```
5. **变到 episodic**：`point_cloud_episodic = transform_points(tf_camera_to_episodic, point_cloud_camera)`。
6. **高度过滤**：`filter_points_by_height(..., min_height=0.61, max_height=0.88)`（默认 HM3D 配置）只留腰高一段，避开地面和天花板。
7. **打到 topdown**：`self._map[py, px] = 1`。
8. **可航行图**：`self._navigable_map = 1 - cv2.dilate(self._map, agent_radius_kernel)`。
9. **fog of war**：`reveal_fog_of_war(...)` 从 agent 当前像素位置按 `topdown_fov`/`max_depth` 画扇形射线，把可见的可航行像素标到 `self.explored_area`，再做一次 dilate。
10. **取连通块**：用 `cv2.findContours` + `pointPolygonTest`，只保留**含有 agent 当前位置的那一块**已探索区，避免被遮挡产生的"孤岛"。
11. **算 frontier**：
    ```python
    explored_area_dilated = cv2.dilate(self.explored_area, 5x5)
    self._frontiers_px = detect_frontier_waypoints(
        self._navigable_map, explored_area_dilated, area_thresh_in_pixels
    )   # 来自 frontier_exploration
    self.frontiers = self._px_to_xy(self._frontiers_px)   # 米
    ```

副作用：写入 `_map / _navigable_map / explored_area / _frontiers_px / frontiers`。

出参：无（in-place）。

举例：步 10 时典型 `obstacle_map.frontiers` 长这样：

```python
np.array([[ 2.45, -1.20],
          [-0.80,  3.10]], dtype=np.float32)  # (2, 2)，episodic 米
```

### 4.3 frontier 选择（核心策略）

回到 `BaseITMPolicy._get_best_frontier`：

1. `_sort_frontiers_by_value(observations, frontiers)`：
   - **V2**：直接走 `ValueMap.sort_waypoints(frontiers, 0.5)`，按 BLIP2 价值排序。
   - **V1（`ITMPolicy`）**：维护一个 `FrontierMap`，给每个 frontier 缓存"第一次看见它时拍的 RGB 算出的 cosine"，避免重复 BLIP2 调用。
   - **V3**：用双通道 value map，按 exploration_thresh 选 reduce 函数。
2. **粘住上一目标**：若上一帧的 `_last_frontier` 还在列表里，且当前价值不差 0.01 以上，就继续走它，避免反复横跳。
3. **AcyclicEnforcer.check_cyclic**：用 `(position, frontier, top_two_values)` 做集合哈希，命中就跳过；防止策略陷入"A→B→A→B"循环。
4. 都被 cyclic 屏蔽就退化为"取离机器人最远的那个"。

输出：`(best_frontier (2,) float32, best_value float)`。

---

## 5. YOLO + Grounding-DINO + MobileSAM：目标点云簇

### 5.1 入口

`BaseObjectNavPolicy.act` 一开始就跑：

```python
object_map_rgbd = self._observations_cache["object_map_rgbd"]
detections = [
    self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
    for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
]
```

`object_map_rgbd` 长度恒为 1（habitat 单相机），所以等价于跑一次 `_update_object_map`。

### 5.2 类别选路：YOLO（COCO）vs Grounding-DINO（非 COCO）

`_get_object_detections(rgb)` 里：

- 如果 `_target_object` 拆开后**任意一个**类是 COCO 类别（80 类，定义在 `vlm/coco_classes.py`），就先用 **YOLOv7** 高精度高阈值（默认 `coco_threshold=0.8`）；
- 否则（MP3D 的 "framed photograph" / "potted plant" / 各种 "table|dining table|..." 等），用 **Grounding-DINO** + caption（拼成 `". ".join(classes)`），低阈值 `non_coco_threshold=0.4`；
- 如果 YOLO 没检到但目标既有 COCO 又有非 COCO 候选（HM3D 不会，MP3D 有），fallback 到 G-DINO 再试一次。
- 检完都会 `filter_by_class(target_classes)` 把和目标无关的类剔掉。

#### YOLOv7Client.predict

```python
class YOLOv7Client:
    def predict(self, image_numpy: np.ndarray) -> ObjectDetections
```

- 入：`image (H,W,3) uint8`（HM3D 默认 (480,640,3) 或 transformer 后的 (224,224,3)）。
- 走 `send_request("http://localhost:12184/yolov7", image=…)`，服务端跑 YOLOv7-E6E。
- 出：`ObjectDetections`：
  ```python
  boxes:   torch.Tensor (K, 4) ∈[0,1]  归一化 xyxy
  logits:  torch.Tensor (K,)            置信度
  phrases: List[str] 长度 K              "chair"/"bed"/...
  ```
- 示例：检到 1 把椅子 → `phrases=["chair"]`, `boxes=tensor([[0.23,0.51,0.41,0.78]])`, `logits=tensor([0.86])`。

#### GroundingDINOClient.predict

跟 YOLO 同形，端口 12181，多接一个 `caption` 入参，例如 `"chair . table . potted plant ."`，输出含 phrases 是匹配上的子串。

### 5.3 MobileSAM.segment_bbox（**关键的精确分割**）

为什么不直接拿 bbox 做点云？因为 bbox 边角会混入背景，反投影出来的点云簇会大量噪点污染 DBSCAN。所以拿 SAM 把 mask 抠出来：

```python
class MobileSAMClient:
    def segment_bbox(self, image: np.ndarray, bbox: List[int]) -> np.ndarray
```

- 入：`image (H,W,3) uint8`；`bbox=[x1,y1,x2,y2]` **整数像素**坐标（bbox_denorm = box * [W,H,W,H]）。
- HTTP 到 12183，服务端 `MobileSAM.segment_bbox` 调 `mobile_sam.SamPredictor`。
- 出：`(H, W) bool` 的 mask（True = 属于该物体）。

`_update_object_map` 里：

```python
for idx in range(len(detections.logits)):
    bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
    object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())  # (H,W) bool

    # （可选）BLIP2 二次确认
    if self._use_vqa:
        annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255,0,0), 2)
        answer = self._vqa.ask(annotated_rgb, "Question: Is this a chair? Answer:")
        if not answer.lower().startswith("yes"): continue

    self._object_masks[object_mask > 0] = 1     # 累计本帧 mask（仅可视化）
    self._object_map.update_map(self._target_object, depth, object_mask,
                                tf_camera_to_episodic, min_depth, max_depth, fx, fy)
```

注意 `_use_vqa` 默认 False（HM3D 不开），开了的话会调 **BLIP2-T5（不是 ITM）** 做 yes/no 问答，端口 12185。

### 5.4 ObjectPointCloudMap.update_map：mask → 3D 点云簇

```python
def update_map(self, object_name, depth_img, object_mask, tf_camera_to_episodic,
               min_depth, max_depth, fx, fy) -> None
```

1. `_extract_object_cloud(depth, mask, ...)`：
   - mask 先 erode（`erosion_size=5`）一圈防边缘溢出。
   - depth 把 0 像素当 "1"（即归一化最远，远到会被 within_range 滤掉），再乘 (max-min)+min 米化。
   - `get_point_cloud(valid_depth, mask, fx, fy)` 反投影：相机系 (M, 3)，x 朝前。
   - `get_random_subarray(cloud, 5000)` 随机降采到 ≤5000 点。
   - `open3d_dbscan_filtering(cloud, eps=0.2, min_points=100)`：DBSCAN（半径 0.2m，至少 100 点），**只保留最大非噪声簇**；若全是噪点则返回空。
2. `transform_points(tf_camera_to_episodic, local_cloud)` → episodic 系点云 (P, 3)。
3. 给每个点拼一个 "within_range" 标志列：
   - 如果 mask 过于偏左/右（`too_offset`：bbox 整个落在左 1/3 或右 1/3 且贴边）→ 全打**随机数** ID，表示"可疑批次"，之后可被 `update_explored` 整批清除。
   - 否则 `within_range = (z <= 0.95 * max_depth)`；不在范围内的 0 也换成同一随机数。
4. `closest_point = _get_closest_point(global_cloud, camera_position)`；若机器人离它太近（<1m）就丢弃这帧（避免太近的不可靠检测）。
5. 拼接到 `self.clouds[object_name]: (∑P, 4)`。

副作用：往 `self.clouds[target]` 累计点云；维度第 4 列是 within_range 编码（1.0 = 可信范围内，其他值 = 同批次随机 id）。

### 5.5 ObjectPointCloudMap.update_explored

每步都跑一次：

```python
def update_explored(self, tf_camera_to_episodic, max_depth, cone_fov) -> None
```

- 取相机当前位置/朝向 + cone_fov（用 `get_fov(fx, depth.shape[1])` 算的水平 FOV）。
- 对每个已存的物体类的点云，看哪些点**现在落在锥形可见区**里（`within_fov_cone(camera, yaw, fov, max_depth*0.5, cloud)`）。
- 如果一批"原本不在 within_range（id 是随机数）"的点现在被看见了，但它的 within_range 还是这个随机 id（说明这帧也没被升格成 1），那就把同 id 整批从地图里删掉——典型场景是远距离误检"看起来像椅子"的桌脚，靠近后没复现就抹掉。

### 5.6 ObjectPointCloudMap.get_best_object

```python
def get_best_object(target_class, curr_position) -> np.ndarray  # (2,)
```

- 优先用 within_range==1 的点子集（如果存在）。
- 找离机器人最近的点 `_get_closest_point`，取 xy。
- 加了滞回：若新最近点离 `self.last_target_coord` 不到 0.1m，或者机器人离它本就 >2m 而差异 <0.5m，就不更新，**避免目标坐标乱跳让 PointNav 重置**。

返回值就是当 act 选 navigate 时喂给 `_pointnav` 的 (x, y) 目标。

---

## 6. 决策：initialize / explore / navigate

`HabitatMixin` + `BaseObjectNavPolicy.act` + `BaseITMPolicy._explore` 拼起来如下：

```python
# 1. HabitatITMPolicyV2.act → HabitatMixin.act → 走子类
# 2. ITMPolicyV2.act 先做 _pre_step（缓存观测）+ _update_value_map（必跑 BLIP2-ITM）
# 3. 跳到 BaseObjectNavPolicy.act：
#    - _update_object_map（YOLO/GDINO+SAM+点云）
#    - goal = self._get_target_object_location(robot_xy)
#    - 三态分支：
#       a) not _done_initializing → _initialize()  # 12 步 30° 原地转 360°
#       b) goal is None           → _explore(observations)
#       c) else                   → _pointnav(goal[:2], stop=True)
```

`_initialize`：

```startLine:endLine:filepath
vlfm/policy/habitat_policies.py
def _initialize(self) -> Tensor:
    self._done_initializing = not self._num_steps < 11
    return TorchActionIDs.TURN_LEFT
```

返回 `tensor([[2]])`，连续 12 步左转 30°（配置里 `turn_angle=30`）。

`_explore`：

```python
def _explore(self, observations):
    frontiers = self._observations_cache["frontier_sensor"]
    if 没 frontier: return self._stop_action  # STOP
    best_frontier, best_value = self._get_best_frontier(observations, frontiers)
    return self._pointnav(best_frontier, stop=False)
```

`_pointnav`：

```python
def _pointnav(self, goal, stop=False) -> Tensor:
    masks = ... # 第 0 步是 0，否则 1
    if 新 goal 离老 goal > 0.1m: self._pointnav_policy.reset(); masks = 0
    rho, theta = rho_theta(robot_xy, heading, goal)   # 米 + 弧度
    if rho < pointnav_stop_radius (=0.9) and stop:    # 只有 navigate 模式才会 STOP
        return self._stop_action
    obs = {
        "depth": image_resize(self._observations_cache["nav_depth"], (224,224), channels_last=True, mode="area"),
        "pointgoal_with_gps_compass": tensor([[rho, theta]], dtype=float32),
    }
    return self._pointnav_policy.act(obs, masks, deterministic=True)
```

`WrappedPointNavResNetPolicy.act` 跑的是 habitat 经典的 ResNet-LSTM PointNav 网络（`data/pointnav_weights.pth`），离散 4 类动作输出。最终：

- 出参：`pointnav_action: Tensor (1, 1) long`，值 ∈ {0,1,2,3}。
- 在 `HabitatMixin.act` 里包成 `PolicyActionData(actions=action, rnn_hidden_states=…, policy_info=[self._policy_info])`，**由 habitat_baselines.worker 取出 actions 灌回 habitat-sim** 执行 STOP/前进/左转/右转，开启下一个 step。

---

## 7. 数据形态速查（值得贴在桌上）

| 阶段 | 张量/数组 | shape | dtype | 典型范围/单位 |
| --- | --- | --- | --- | --- |
| habitat-sim 出 obs | `observations["rgb"]` | (1, H, W, 3) | uint8 | 0–255 |
|  | `observations["depth"]` | (1, H, W, 1) | float32 | 0–1（再 *max+min 米化） |
|  | `observations["gps"]` | (1, 2) | float32 | 米，y 西负 |
|  | `observations["compass"]` | (1, 1) | float32 | 弧度 |
|  | `observations["objectgoal"]` | (1, 1) | int64 | HM3D 0–5 / MP3D 0–20 |
| 缓存 | `_observations_cache["robot_xy"]` | (2,) | float32 | 米 |
|  | `tf_camera_to_episodic` | (4, 4) | float64 | 米 + 弧度 |
| BLIP2-ITM | `cosine` | scalar | float | 大致 [0, 1]，0.5 起步算"像" |
| ValueMap | `_value_map` | (1000, 1000, C) | float32 | 0–1，C=1 或 2 |
|  | `_map`(confidence) | (1000, 1000) | float32 | 0–1 |
| ObstacleMap | `_map`/`_navigable_map`/`explored_area` | (1000, 1000) | bool | True/False |
|  | `frontiers` | (N, 2) | float32 | episodic 米 |
| YOLO/GDINO | `ObjectDetections.boxes` | (K, 4) | float | 归一化 xyxy ∈[0,1] |
|  | `.logits` | (K,) | float | 0–1 |
|  | `.phrases` | list[str] K | str | 类别名 |
| MobileSAM | `object_mask` | (H, W) | bool | True 物体内 |
| 物体点云 | `clouds[obj]` | (P, 4) | float32 | 列：x,y,z(米), within_range_id |
| PointNav 输入 | `depth` | (1, 224, 224, 1) | float32 | 0–1 |
|  | `pointgoal_with_gps_compass` | (1, 2) | float32 | (rho 米, theta 弧度) |
| PointNav 输出 | `action` | (1, 1) | long | 0:STOP 1:FWD 2:L 3:R |

---

## 8. 一次 act() 的"真实顺序"（避免被论文图误导）

按代码实际跑的顺序（HabitatITMPolicyV2 + 默认配置）：

1. `HabitatMixin.act` 把 `objectgoal` id 翻译成类名字符串塞进 `obs_dict`。
2. 调父类 `ITMPolicyV2.act`：
   1. `_pre_step` → `_cache_observations`（**这里如果 `_compute_frontiers=True`，会先跑一次 ObstacleMap.update_map 并取 frontiers**）。
   2. `_update_value_map`（BLIP2-ITM 一次/多次 + ValueMap.update_map）。
   3. 调祖父类 `BaseObjectNavPolicy.act`：
      1. `_update_object_map`：YOLO 或 G-DINO 检测 → 每个 bbox 跑 MobileSAM → （可选 BLIP2-VQA 复核）→ 反投影点云 + DBSCAN → 累计到 `ObjectPointCloudMap`。
      2. `goal = self._get_target_object_location(robot_xy)`：若已有目标点云，就拿最近点。
      3. 分支决策（initialize / explore / navigate）。
      4. explore 分支再回到 `BaseITMPolicy._explore` → `_get_best_frontier` → `_pointnav(stop=False)`。
3. 返回 `(action_tensor, rnn_hidden_states)`，再被 `HabitatMixin.act` 包成 `PolicyActionData` 给到 habitat_baselines。

**核心一句话**：ValueMap 永远在 ObjectMap 之前更新；ObstacleMap 在 `_cache_observations` 内顺手更新（早于 ValueMap）；frontier 检测、BLIP2 评分、YOLO/SAM 全在出动作之前完成；最后用 `WrappedPointNavResNetPolicy.act` 输出离散动作。

---

## 9. VLM 服务端口一览

启动脚本 `scripts/upstream/launch_vlm_servers.sh` 会开 5 个独立 Flask 进程：

| 端口（默认 / 环境变量） | 服务 | 客户端类 | 作用 |
| --- | --- | --- | --- |
| 12181 (`GROUNDING_DINO_PORT`) | Grounding-DINO | `GroundingDINOClient` | 非 COCO 类别开放词检测 |
| 12182 (`BLIP2ITM_PORT`) | BLIP2-ITM | `BLIP2ITMClient` | 图文余弦 → ValueMap |
| 12183 (`SAM_PORT`) | MobileSAM | `MobileSAMClient` | bbox → 精细 mask |
| 12184 (`YOLOV7_PORT`) | YOLOv7-E6E | `YOLOv7Client` | COCO 类别检测 |
| 12185 (`BLIP2_PORT`) | BLIP2-T5 | `BLIP2Client` | （可选）VQA 复核 |

通信方式：base64 JPEG（`image_to_str` 90% 质量）+ JSON POST，重试 10 次（`server_wrapper.send_request`）。这就是为啥 VLFM 在主进程崩了 VLM 也能撑住——客户端纯无状态。

---

## 10. 想自己 trace 一帧时的入口建议

- 想知道"我这一步的 rgb/depth/gps 真长啥样"：在 `vlfm/policy/habitat_policies.py` 的 `_cache_observations` 第 181-189 行打断点，print 各 shape。
- 想看 BLIP2 给的分数：在 `vlfm/policy/itm_policy.py` 的 `_update_value_map` 里 print `cosines`。
- 想看 frontier：`vlfm/mapping/obstacle_map.py` 的 `update_map` 末尾 print `self.frontiers`。
- 想看物体点云：`vlfm/mapping/object_point_cloud_map.py` 的 `update_map` 里 print `global_cloud.shape` 和 `self.clouds[object_name].shape`。
- 想看最终动作：`vlfm/policy/base_objectnav_policy.py` 的 `act` 已经 print 了 `Step / Mode / Action`。

---

## 附 A：和已有笔记的对应

- 总览/直觉对账 → [00_总览与直觉核对.md](./00_总览与直觉核对.md)
- BLIP2/ValueMap 细节（融合公式推导） → [01_BLIP2_ValueMap_推理与融合.md](./01_BLIP2_ValueMap_推理与融合.md)
- YOLO + G-DINO + SAM 协同 → [02_YOLO_Grounding_SAM_对象定位.md](./02_YOLO_Grounding_SAM_对象定位.md)
- ObstacleMap / Frontier → [03_ObstacleMap_与_Frontier.md](./03_ObstacleMap_与_Frontier.md)
- act / explore / navigate 时序 → [04_决策流程_act_explore_navigate.md](./04_决策流程_act_explore_navigate.md)
- 形状速查 → [05_数据形态速查表.md](./05_数据形态速查表.md)
- Habitat 仿真接口（含 sensor/action 注册细节） → [06_仿真环境_Habitat与接口.md](./06_仿真环境_Habitat与接口.md)

---

## 附 B：模块小节子文档（用 HTML 画图，更清楚）

> 下面这些 HTML 子文档是对本 README 中"一句话带过"的几块的展开，含 SVG 网络架构图 / 决策树 / 算法分步示意图。
> 直接在浏览器里打开（双击文件即可），或者用 IDE 的 HTML 预览插件。

| 子文档 | 对应 README 小节 | 主要内容 |
| --- | --- | --- |
| [sub_01_PointNav_ResNet内部.html](./sub_01_PointNav_ResNet内部.html) | §6 末尾 `WrappedPointNavResNetPolicy` | depth → ResNet-18(GN, base_planes=32) → compression Conv → Flatten 2048 → Linear 512 → LSTM(576→512)×2 → Categorical(4) / Gaussian(2)。含两张 SVG 架构图、三套加载分支对照表、(h,c) reset 的时序、推理时断点位置 |
| [sub_02_fog_of_war算法.html](./sub_02_fog_of_war算法.html) | §4.2 第 9 步 `reveal_fog_of_war` | 不是逐根 raycast，而是"画扇形 → 找障碍 contour → 取最外两切线 → 撕碎扇形 → 取含 agent 那块"。含 5 步示意 SVG、坐标轴 / 角度变换的常见踩坑、参数对 frontier 数量的敏感性 |
| [sub_03_ITMPolicyV3_双通道.html](./sub_03_ITMPolicyV3_双通道.html) | §3.1 末尾 / §4.3 第 1 步 V3 | 两个 prompt（target + exploration）→ 双通道 ValueMap → `_reduce_values` 按 `max(target_values)` vs `exploration_thresh` **硬切换**到 target 通道或 exploration 通道。含数据流 SVG、决策树 SVG、阈值挑选建议、典型失败模式 |

> 还想追加哪一块？候选清单（每条都能再起一份子文档）：
> - **AcyclicEnforcer** 的 `(robot_xy, frontier, top_two_values)` 哈希实现细节、循环判定的边界条件
> - **DBSCAN（open3d_dbscan_filtering）** 在物体点云上的参数选择，以及"过远批次"为什么用随机 id 标记而不是单调 id
> - **`_cache_observations`** 里 `compute_frontiers=True` 跟 habitat 内置 `frontier_sensor` 两条路的输出差异
> - **HabitatMixin.from_config** 跟 Hydra config 之间的注入链（`baseline_registry`、`ConfigStore.store(node=VLFMPolicyConfig)`）
> - **`HM3D` vs `MP3D`** 数据集差异对 prompt / threshold / fx 的具体影响
