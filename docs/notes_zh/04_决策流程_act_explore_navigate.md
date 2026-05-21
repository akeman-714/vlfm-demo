# 04 · 决策流程：一次 `act()` 里到底发生什么

> 关键文件：
> - `vlfm/policy/base_objectnav_policy.py:BaseObjectNavPolicy.act`
> - `vlfm/policy/itm_policy.py:BaseITMPolicy._explore / _get_best_frontier`
> - `vlfm/policy/habitat_policies.py:HabitatMixin.act / _initialize / _cache_observations`

## 4.1 调用层级（从外到内）

```
HabitatMixin.act (override 解析 objectgoal id→name)
  └─ super().act → ITMPolicyV2.act
        ├─ _pre_step (reset?, cache observations)
        ├─ _update_value_map           ◄── BLIP2-ITM 余弦 + ValueMap 更新
        └─ BaseObjectNavPolicy.act
              ├─ _pre_step (idempotent)
              ├─ _update_object_map    ◄── YOLO/Grounding + SAM + ObjectPointCloudMap
              ├─ goal = _get_target_object_location(robot_xy)
              ├─ if not _done_initializing: _initialize()        # 0~10步原地转 30°×12
              │  elif goal is None:        _explore(obs)          # ← ValueMap 选 frontier
              │  else:                     _pointnav(goal, stop=True)  # ← 直奔目标
              └─ _get_policy_info(...)（可视化信息）
```

> 注意：`_pre_step` 在 V2 / V3 里被调用 **两次**（V2 自身一次、父类一次）。`_cache_observations` 用 `len(self._observations_cache) > 0` 当门闩，第二次会直接跳过。

## 4.2 第一阶段：原地转身（initialize）

`HabitatMixin._initialize`（`habitat_policies.py:150-153`）

```python
def _initialize(self) -> Tensor:
    self._done_initializing = not self._num_steps < 11  # 转 12 次
    return TorchActionIDs.TURN_LEFT
```

- 配合 `base_explorer.turn_angle=30`（yaml 里设置），机器人原地左转 `12 × 30° = 360°` 建图。
- 这期间 ValueMap、ObstacleMap、ObjectPointCloudMap 都在更新。
- Reality 里换成"机械臂横扫"`INITIAL_ARM_YAWS = [-90,-60,-30,0,30,60,90,0]` 八个角度。

## 4.3 第二阶段：检测覆盖判定（YOLO/Grounding "高于" BLIP2 的真正含义）

`BaseObjectNavPolicy.act:122-138`

```python
object_map_rgbd = self._observations_cache["object_map_rgbd"]
detections = [
    self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
    for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
]
robot_xy = self._observations_cache["robot_xy"]
goal = self._get_target_object_location(robot_xy)

if not self._done_initializing:
    mode = "initialize"
    pointnav_action = self._initialize()
elif goal is None:
    mode = "explore"
    pointnav_action = self._explore(observations)
else:
    mode = "navigate"
    pointnav_action = self._pointnav(goal[:2], stop=True)
```

- `_update_object_map` 每步都跑（YOLO/Grounding+SAM+点云累积）。
- `_get_target_object_location` 查 `ObjectPointCloudMap.has_object(target)`：
  - **没目标点云** → `goal=None` → 走 explore 用 ValueMap；
  - **有目标点云** → `goal=(x,y)` → 走 pointnav 直奔最近点。

📌 **这就是"YOLO/Grounding 高于 BLIP2"的真实机制**：不是覆盖某张图，而是 if-else 决定用哪条目标。

> 一旦进入 navigate 模式，下一步如果 YOLO/Grounding 仍能看到目标，`get_best_object` 的"抖动抑制"会让目标点不剧烈变化；如果突然看不到了（如转弯遮挡），但 `ObjectPointCloudMap.clouds[target]` 里之前累积的点没有被 `update_explored` 清掉，依然返回 navigate 模式。只有点云被全部清空才会回 explore。

## 4.4 第三阶段：explore — 用 ValueMap 选 frontier

`BaseITMPolicy._explore` 和 `_get_best_frontier`（`itm_policy.py:64-152`）

```python
def _explore(self, observations):
    frontiers = self._observations_cache["frontier_sensor"]
    if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
        return self._stop_action
    best_frontier, best_value = self._get_best_frontier(observations, frontiers)
    pointnav_action = self._pointnav(best_frontier, stop=False)
    return pointnav_action
```

`_get_best_frontier` 的策略：

1. `_sort_frontiers_by_value(observations, frontiers)`
   - **V2**：`ValueMap.sort_waypoints(frontiers, 0.5)` 直接按 value（半径 0.5 m 内的中位数）降序。
   - **V3**：附加 `reduce_fn=_reduce_values`，先看 max(target_channel)，没过阈值就降级用 exploration_channel。
   - **ITMPolicy (V1)**：用 `FrontierMap` 而不是 ValueMap，每个 frontier 第一次进表时单独 BLIP2 算一次 cosine。
2. 优先黏住上次的 frontier（如果还在列表里、value 没差超 0.01）。
3. 用 `AcyclicEnforcer` 跳过 (pos, frontier, top2values) 三元组已访问过的循环点。
4. 都循环就挑离自己最远的那个（保守探索）。

`AcyclicEnforcer`(`vlfm/policy/utils/acyclic_enforcer.py`) 就是个 `set[hash(pos, action, top2)]`。

## 4.5 PointNav：把 (x,y) 转成动作

`BaseObjectNavPolicy._pointnav`（`base_objectnav_policy.py:243-279`）

```python
masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
if not np.array_equal(goal, self._last_goal):
    if np.linalg.norm(goal - self._last_goal) > 0.1:
        self._pointnav_policy.reset()
        masks = torch.zeros_like(masks)
    self._last_goal = goal
robot_xy = self._observations_cache["robot_xy"]
heading = self._observations_cache["robot_heading"]
rho, theta = rho_theta(robot_xy, heading, goal)
rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
obs_pointnav = {
    "depth": image_resize(self._observations_cache["nav_depth"], (224, 224), ...),
    "pointgoal_with_gps_compass": rho_theta_tensor,
}
if rho < self._pointnav_stop_radius and stop:
    self._called_stop = True
    return self._stop_action
action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
return action
```

- 用 Habitat 训好的 **ResNet50-LSTM PointNav 策略**（`data/pointnav_weights.pth`）。
- 输入 `(224,224)` depth + `(ρ, θ)` 极坐标 → 输出离散动作 (STOP / FWD / TL / TR) 或 Spot 的连续 (linear, angular)。
- 切换 goal 超过 0.1 m 时重置 LSTM 隐状态。
- 距离 < `0.9 m`（HM3D 默认）且 `stop=True` 时调 STOP。

## 4.6 _cache_observations：观测打包

`HabitatMixin._cache_observations`（`habitat_policies.py:173-237`）只在一帧内**第一次** `_pre_step` 时执行：

```python
self._observations_cache = {
    "frontier_sensor": frontiers,                      # (N, 2) 米
    "nav_depth": observations["depth"],                # 给 pointnav 的 raw depth tensor
    "robot_xy": robot_xy,                              # (2,) 米
    "robot_heading": camera_yaw,                       # 标量 弧度
    "object_map_rgbd": [(rgb, depth, tf, mind, maxd, fx, fy)],
    "value_map_rgbd": [(rgb, depth, tf, mind, maxd, camera_fov)],
    "habitat_start_yaw": ...,
}
```

注意：
- `depth` 是经 `depth_camera_filtering.filter_depth` 过的（去玻璃噪声）；
- `value_map_rgbd` 用 fov（弧度），`object_map_rgbd` 用 fx/fy；
- list 包装是为了支持多相机（Reality 里 Spot 有 5 个相机，一帧里同时更新 ObstacleMap）。

ITMPolicyV2 自己的 `_pre_step` 多一步：

```python
class ITMPolicyV2(BaseITMPolicy):
    def act(self, ...):
        self._pre_step(observations, masks)
        self._update_value_map()              # ← 这里走 BLIP2-ITM + ValueMap
        return super().act(...)
```

所以**真实先后**：

```
1. _pre_step → cache 一帧 obs（含 ObstacleMap 更新 + frontier 计算）
2. _update_value_map → 跑 BLIP2 cosines + ValueMap.update_map
3. _update_object_map（在 super().act 内）→ 跑 YOLO/Grounding + SAM + 点云累积
4. _get_target_object_location → 决定 explore / navigate
5. 选 frontier 或直奔目标
6. _pointnav 输出动作
```

> 你说"三者同时跑"——从 wall-clock 上看，它们都是**串行**调用三个独立的 Flask 服务（BLIP2_ITM @ 12182、YOLOv7 @ 12184、GroundingDINO @ 12181、SAM @ 12183），不同步但顺序固定。

## 4.7 ITMPolicy 三个版本的差异

| Policy | ValueMap 通道数 | BLIP2 调用 | frontier 排序 |
| --- | --- | --- | --- |
| `ITMPolicy` (V1) | 1（只可视化用） | 每个 frontier 用各自首次见到时的余弦 | `FrontierMap.sort_waypoints` |
| `ITMPolicyV2` | 1 | 每步对当前 rgb 算一次余弦 | `ValueMap.sort_waypoints` 单通道 max |
| `ITMPolicyV3` | 2 | 每步对当前 rgb 算两次（target prompt + exploration prompt） | `ValueMap.sort_waypoints` + `_reduce_values` 切换 |

HM3D 默认 yaml `name: "HabitatITMPolicyV2"`。

## 4.8 一次完整 act 时序图

```
t=k 开始
│
├── HabitatMixin.act (objectgoal id→name)
│
├── ITMPolicyV2.act
│     │
│     ├── _pre_step
│     │     ├── (若 mask=0) _reset()
│     │     └── _cache_observations  ► ObstacleMap.update_map(depth)
│     │                                ► frontiers = ObstacleMap.frontiers
│     │                                ► self._observations_cache[...] 装好
│     │
│     ├── _update_value_map
│     │     ├── for prompt in text_prompts:
│     │     │       cosine = BLIP2ITMClient.cosine(rgb, prompt)
│     │     └── ValueMap.update_map(cosines, depth, tf, fov)
│     │           ├── _process_local_data(depth, fov, min, max)        ── (201,201) c_curr
│     │           ├── _localize_new_data → curr_map (1000,1000)
│     │           └── _fuse_new_data → 更新 _map, _value_map
│     │
│     └── BaseObjectNavPolicy.act
│           ├── _pre_step (跳过：cache 已存在)
│           ├── _update_object_map
│           │     ├── _get_object_detections(rgb)
│           │     │     ├── YOLO 或 Grounding
│           │     │     └── filter_by_class / filter_by_conf
│           │     ├── for each detection:
│           │     │     ├── SAM.segment_bbox → mask (H,W)
│           │     │     ├── (optional) BLIP2 VQA confirm
│           │     │     └── ObjectPointCloudMap.update_map(...)
│           │     └── ObjectPointCloudMap.update_explored(...)
│           │
│           ├── goal = ObjectPointCloudMap.get_best_object(target, robot_xy) or None
│           │
│           ├── if init not done: TURN_LEFT
│           │   elif goal is None: _explore()
│           │   │     ├── frontiers = cache["frontier_sensor"]
│           │   │     ├── ValueMap.sort_waypoints(frontiers, 0.5)
│           │   │     ├── 处理黏滞 + AcyclicEnforcer
│           │   │     └── _pointnav(best_frontier, stop=False)
│           │   else: _pointnav(goal, stop=True)
│           │
│           ├── _pointnav
│           │     ├── rho_theta(robot_xy, heading, goal)
│           │     ├── 若 rho < 0.9 & stop=True: STOP
│           │     └── PointNavResNetPolicy.act(depth(224,224), rho_theta) → action
│           │
│           └── _get_policy_info(detections[0]) → 可视化
│
└── 输出 action tensor 给 habitat env
```

## 4.9 你直觉对应到代码的对照

| 直觉 | 对应代码段 |
| --- | --- |
| "BLIP2 看到相似物体" | `BLIP2ITMClient.cosine(rgb, prompt)` |
| "投到 2D 锥形" | `ValueMap._process_local_data` → (201,201) `visible_mask` |
| "重合按公式融合" | `ValueMap._fuse_new_data` → `use_max_confidence` 分支或加权平均 |
| "走向最近点" | `ObjectPointCloudMap.get_best_object` → `_get_closest_point` |
| "YOLO/Grounding 同时跑" | 实际是 **二选一**：has_coco→YOLO，否则 Grounding，兜底再 Grounding |
| "YOLO/Grounding 层级高于 BLIP2" | `if goal is None: explore() else: pointnav(goal)` |
| "覆盖下一个走向点" | 不是覆盖 ValueMap，是直接跳过 explore 分支 |
