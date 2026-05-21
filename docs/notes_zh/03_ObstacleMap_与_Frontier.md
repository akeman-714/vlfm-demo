# 03 · ObstacleMap：障碍、已探索、可航行、Frontier

> 关键文件：
> - `vlfm/mapping/obstacle_map.py`
> - `vlfm/mapping/base_map.py`
> - 第三方：`frontier_exploration.frontier_detection.detect_frontier_waypoints`、`frontier_exploration.utils.fog_of_war.reveal_fog_of_war`

ObstacleMap 跟 ValueMap 是**并列的**两张俯视图，共享 `size=1000, pixels_per_meter=20` 的坐标系（50 m × 50 m）。

## 3.1 三张二值图

```python
class ObstacleMap(BaseMap):
    _map           : (1000, 1000) bool   # 障碍栅格
    explored_area  : (1000, 1000) bool   # 雾战已揭开的区域
    _navigable_map : (1000, 1000) bool   # _map 的反 + 机器人半径膨胀
    frontiers      : (N, 2) float        # 当前帧 frontier 的 xy (米)
    _frontiers_px  : (N, 2) int          # 像素坐标
```

## 3.2 update_map 流程

`vlfm/mapping/obstacle_map.py:55-153`

### 3.2.1 障碍点云

```86:101:vlfm/mapping/obstacle_map.py
if update_obstacles:
    if self._hole_area_thresh == -1:
        filled_depth = depth.copy()
        filled_depth[depth == 0] = 1.0
    else:
        filled_depth = fill_small_holes(depth, self._hole_area_thresh)
    scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
    mask = scaled_depth < max_depth
    point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
    point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
    obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)
    xy_points = obstacle_cloud[:, :2]
    pixel_points = self._xy_to_px(xy_points)
    self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1
```

- 把 depth 反归一化到米；空洞补 1（远）；
- 用 `get_point_cloud` 算出相机系下点云；
- 变到 episodic frame；
- 按高度过滤：`min_height=0.61, max_height=0.88`（默认 VLFM 配置，过滤天花板和地面）；
- 投影到 `(x, y)`，写进 `_map`。

### 3.2.2 可航行图（机器人半径膨胀）

```103:109:vlfm/mapping/obstacle_map.py
self._navigable_map = 1 - cv2.dilate(
    self._map.astype(np.uint8),
    self._navigable_kernel,
    iterations=1,
).astype(bool)
```

- `agent_radius = 0.18 m` → 膨胀核大小 `int(0.18·20·2)=7`（圆整到奇数）。
- 把障碍向外胀 7 px，机器人 footprint 就被包进去，反过来 = 可航行区。

### 3.2.3 雾战（已探索）

```115:127:vlfm/mapping/obstacle_map.py
agent_xy_location = tf_camera_to_episodic[:2, 3]
agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]
new_explored_area = reveal_fog_of_war(
    top_down_map=self._navigable_map.astype(np.uint8),
    current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
    current_point=agent_pixel_location[::-1],
    current_angle=-extract_yaw(tf_camera_to_episodic),
    fov=np.rad2deg(topdown_fov),
    max_line_len=max_depth * self.pixels_per_meter,
)
new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
self.explored_area[new_explored_area > 0] = 1
self.explored_area[self._navigable_map == 0] = 0
```

- `reveal_fog_of_war` 来自第三方 `frontier_exploration` 包，做的是从机器人当前位置发射射线（受可航行图阻挡），把击中范围标 1。
- `topdown_fov` 是把相机 hfov 投影到 2D 后的角度：

```python
# habitat_policies.py 里这个值就是 camera_fov_rad
# 因为 depth 已经压成一行，水平 fov 直接当 2D fov 用
```

- 后处理：取**包含 agent 的最大连通区**，过滤掉孤岛误探索。

### 3.2.4 Frontier 检测

```155:169:vlfm/mapping/obstacle_map.py
def _get_frontiers(self) -> np.ndarray:
    explored_area = cv2.dilate(
        self.explored_area.astype(np.uint8),
        np.ones((5, 5), np.uint8),
        iterations=1,
    )
    frontiers = detect_frontier_waypoints(
        self._navigable_map.astype(np.uint8),
        explored_area,
        self._area_thresh_in_pixels,
    )
    return frontiers
```

- 用 `frontier_exploration` 的 `detect_frontier_waypoints`：
  - **frontier** = 已探索 - 未探索的边界（且必须在 navigable 内）；
  - `area_thresh = 1.5 m²` → `1.5 · 20² = 600 px²`，太小的 frontier 段被忽略；
  - 输出 `(N, 2)` 整数像素坐标。
- 在外面用 `_px_to_xy` 转米。

## 3.3 ObstacleMap 与 ValueMap 的协作

`ITMPolicyV2/V3` 默认 `sync_explored_areas=False`，**两张图独立累积**。但 ValueMap 在可视化 / 选 frontier 时会参考 ObstacleMap：

- `ValueMap.visualize(obstacle_map=...)` 把未探索区域涂白；
- `sort_waypoints` 的输入 frontier 就来自 `ObstacleMap.frontiers`；
- 如果 `sync_explored_areas=True`（Reality 默认），ValueMap 的 `_fuse_new_data` 开头就用 `ObstacleMap.explored_area==0` 把 ValueMap 那块清 0。

## 3.4 数据形状速查

| 变量 | shape | dtype | 单位 / 含义 |
| --- | --- | --- | --- |
| `depth` 输入 | (H, W) | float32 | 归一化 [0,1] |
| `scaled_depth` | (H, W) | float32 | 米，min_depth~max_depth |
| `point_cloud_camera_frame` | (M, 3) | float32 | 相机系 (z, -x, -y) |
| `point_cloud_episodic_frame` | (M, 3) | float32 | episodic xyz |
| `obstacle_cloud` | (M', 3) | float32 | 高度 ∈ [0.61, 0.88] |
| `self._map` | (1000, 1000) | bool | 1=障碍 |
| `self._navigable_map` | (1000, 1000) | bool | 1=可走 |
| `self.explored_area` | (1000, 1000) | bool | 1=已揭开 |
| `self._frontiers_px` | (N, 2) | int | px (row, col) |
| `self.frontiers` | (N, 2) | float | 米 |

## 3.5 可视化（视频时）

`ObstacleMap.visualize` 输出 `(1000, 1000, 3) uint8`：
- 白底
- 已探索浅绿 `(200, 255, 200)`
- 不可航行灰 `(100, 100, 100)`
- 障碍黑
- frontier 蓝圈半径 5px
- 叠加机器人轨迹

`ValueMap.visualize` 用 Inferno colormap 上色，frontier 蓝、目标绿、当前选中黄。两张图通常在 habitat_visualizer 里拼接。
