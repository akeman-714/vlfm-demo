# 模块 3：全局导航（A\* 找路 + 控制器走路）

**改动量**：中（不是大）　**依赖**：`skimage`(已装) + `scipy`(已装)　**前置模块**：无（可独立先做）　**开关**：`VLFM_GLOBAL_NAV=1`

## 目标

把 act() 的 navigate 分支从「直接奔点 `_pointnav(goal, stop=True)`」升级为
「在持久化/live 障碍图上 **A\* 算路 → 控制器逐 waypoint 走 → 到点 STOP**」。
拆两半：**(a) 找路**（唯一要新写的核心，但有现成库）+ **(b) 走路**（~20 行几何控制器）。

## 交付物 & 验收标准

> 先用「回出生点 `(0,0)`」验收，**不依赖模块 1/2**。

- [ ] **验收 1（最难、最先）**：`VLFM_GLOBAL_NAV=1` + 临时把 navigate 目标设成 `(0,0)`，跑 cat_demo →
      机器人**绕开墙体回到出生点并 STOP**，不卡墙、不直线穿障碍。视频可见路径绕行。
- [ ] **验收 2（目标在障碍上）**：目标设成一个落在障碍像素上的点 → 不报错，snap 到最近可走点并在 `pointnav_stop_radius` 内停下。
- [ ] **验收 3（不连通/未探索）**：目标设在尚未探索区域 → A\* 给乐观路（未见=可走），边走边 replan；真被已知障碍围死 → 回落 frontier 探索，不崩。
- [ ] **验收 4（回归）**：`VLFM_GLOBAL_NAV` 未设时，navigate 分支保持原 `_pointnav(goal, stop=True)`，结果与现状一致。
- [ ] **联调（接模块 2）**：navigate-from-memory 态——reset 后 `_remembered_goal` 命中 → 同一套 `_navigate_global` 直奔记忆猫点。

### 如何跑验收（已落地的开关，无需改代码）

验收 1/2/3 都用调试钩子 `VLFM_NAV_DEBUG_GOAL="x,y"` 强制 navigate 到指定 episodic 点；
`VLFM_NAV_DEBUG_AFTER=N` 让机器人**先探索 N 步再回奔**（否则出生点就在 (0,0)，会立刻 STOP，看不到绕行）。

```bash
# 验收 1（回家绕墙）：先探索 40 步，再 A* 回出生点 (0,0)，保守图（只走已探索空地）
VLFM_GLOBAL_NAV=1 VLFM_NAV_DEBUG_GOAL=0,0 VLFM_NAV_DEBUG_AFTER=40 \
  bash scripts/cat_demo/eval_cat_demo.sh
# 验收 2（目标压障碍上）：把 DEBUG_GOAL 设到一个已知墙体里的点，应 snap+STOP，不报错
# 验收 3（未探索/不连通）：把 DEBUG_GOAL 设到远处未探索点，乐观给路、边走边 replan
# 验收 4（回归）：不设 VLFM_GLOBAL_NAV → navigate 分支仍走原 _pointnav(goal, stop=True)
```

> 已通过：`test/test_path_planning.py`（adapter 往返 + A\* 绕墙 + snap + 围死回 None + 抽稀，8 项）
> 与一个闭环仿真（真 `_navigate_global` 驱动假 self，开阔/绕墙两例均到点 STOP）。
> 验收 1–4 需带 VLM 服务的整跑，留待 GPU run。

## 子模块 (a)：找路 — 新建 `vlfm/utils/path_planning.py`

| 函数 | 实现 | 依赖 |
|---|---|---|
| `plan_path(navigable, start_px, goal_px) -> list[px] | None` | `skimage.graph.route_through_array(cost, start, end, geometric=True)`；`cost = where(navigable, 1, BIG)` | skimage 0.24.0 ✓ |
| `snap_to_navigable(navigable, px) -> px` | `scipy.ndimage.distance_transform_edt(~navigable, return_indices=True)` 取最近可走像素 | scipy 1.13.1 ✓ |
| `downsample_path(path_px, ...) -> list[px]` | 按共线/每 N 像素抽稀 waypoint | numpy |

- **坐标约定**：A\* 在数组里按 `(row, col)`；与 `obstacle_map.py:101` 的 `_map[px[:,1], px[:,0]]` 一致。
  xy↔px **只走** `obstacle_map._xy_to_px` / `_px_to_xy`（`base_map.py:35-60`），写一个薄 adapter，**禁止重写翻转**。
- `route_through_array` 返回 `(indices_list, cost)`；`indices_list` 是 `[(r,c), ...]`，注意是 (row,col)。

> ⚠️ **已被模块 4 Part A 取代**：geometric 跟随器与 `waypoint_controller.py` 已删除，全局导航统一走 A\* + PointNav follower（pointnav 自带局部避障，不再需要卡死/碰撞补丁）。下文为历史设计存档。

## 子模块 (b)：走路 — 新建 `vlfm/policy/utils/waypoint_controller.py`（选项 B，推荐）

```python
def step_towards(waypoint_xy, robot_xy, heading, turn_angle_rad, arrive_radius):
    rho, theta = rho_theta(robot_xy, heading, waypoint_xy)   # 复用 geometry_utils.rho_theta
    if rho < arrive_radius:        return None                # 到点 -> 上层 advance/STOP
    if abs(theta) > turn_angle_rad/2:
        return TURN_LEFT if theta > 0 else TURN_RIGHT
    return MOVE_FORWARD
```

- 动作用 `TorchActionIDs`（`vlfm/policy/habitat_policies.py:66-70`），shape `[[x]]` 与 `_stop_action` 一致，
  下游 `base_objectnav_policy.py:140` 的 `.detach().cpu().numpy()[0]` 直接消费。
- ⚠️ **转角/步长必须对齐 sim**：`eval_cat_demo.sh` 设了 `base_explorer.turn_angle=30`；Habitat 前进默认 0.25m。
  控制器的 `turn_angle_rad`、`arrive_radius` **从 config/常量读，别硬编码**，否则角度模型与实际步进对不上、原地抖。

### 选项 A（备选，要学习式局部避障时）
把 next waypoint 喂现成 `_pointnav`（`base_objectnav_policy.py:243-279`），它输出合法动作带局部避障。
代价：目标变化 >0.1m 会 reset RNN（`:256-259`）；每帧挪 waypoint 会反复重置。
→ 缓解：lookahead，只在进入 `arrive_radius` 才 advance，advance 间 goal 固定。**cat_demo 静态场景默认用选项 B。**

## 集成 — 改 `act()` 的 navigate 分支

落点 `vlfm/policy/base_objectnav_policy.py:136-138`：

```python
else:
    mode = "navigate"
    if os.environ.get("VLFM_GLOBAL_NAV") == "1":
        pointnav_action = self._navigate_global(goal[:2])     # 新方法
    else:
        pointnav_action = self._pointnav(goal[:2], stop=True) # 原行为（回归保护）
```

- 新方法 `self._navigate_global(goal_xy)`：plan(缓存) → 取 next waypoint → `step_towards` → 到点 STOP；含 replan。
- 落在 `BaseObjectNavPolicy`，`HabitatITMPolicyV2.act`（`itm_policy.py:251-261`）走 `super().act()` 自动继承，**单一落点**。
- **navigate-from-memory 态**（接模块 2）：在 `act()` 分支判断里加一支——
  目标未见（`goal is None`）但 `self._remembered_goal is not None` → `self._navigate_global(self._remembered_goal)`；
  一旦 `_get_target_object_location` 非 None，切回真实 goal。
- `_explore`（`itm_policy.py:64-74`）维持原样。

### 新增状态（`__init__` 建，`_reset()` 清）
`self._global_path`（waypoint px 列表）、`self._path_goal`（算路时的 xy）、`self._waypoint_idx`。

### Replan 触发（别每帧重算）
- `self._global_path is None`（首次/reset 后）
- 到达当前 waypoint → advance；到最后一个 → 进入 STOP 收尾
- goal 漂移 > 阈值（物体点随点云更新会动）
- 下一 waypoint 被新观测变成 non-navigable（撞上新见障碍）
- 每 M≈30 步安全重算（吸收新障碍）

## 边界情况（要处理的真坑）

| 坑 | 解法 | 依据 |
|---|---|---|
| **目标压在障碍上**（猫贴墙/家具上，px navigable=0）→ A\* 到不了 | `snap_to_navigable` 收尾，靠 `_pointnav_stop_radius` 停 | `base_objectnav_policy.py:73,275` |
| **中间未探索→不连通** | navigable 把**未见=可走**（`obstacle_map.py:105-109`：navigable=非膨胀障碍，未见处障碍=0）→ A\* 乐观给「假设空旷」路，边走边 replan；真被已知障碍围死才回落 frontier | `obstacle_map.py:105-109` |
| **回家/回记忆点要稳** | 用**保守图** `_navigable_map & explored_area`（只走已知空地）；新鲜冲目标用**乐观图** navigable。`_navigate_global` 加 `conservative: bool` 参 | 同上 |
| **起点自身 non-navigable**（机器人被膨胀进障碍） | start 也 `snap_to_navigable` | — |
| **轴/翻转 off-by-one** | 一个 tested adapter 统一 (row,col)↔xy，写单测 | `obstacle_map.py:101` |
| 窄道 vs 体宽 | navigable 已按 `agent_radius=0.18` 膨胀，A\* 天然避让，无需额外处理 | `obstacle_map.py:43-46` |
| 振荡/反复 replan | 仅按上面触发器 replan；必要时复用 `AcyclicEnforcer`（`vlfm/policy/utils/acyclic_enforcer.py`）风格守卫 | `itm_policy.py:130` |
| 越界 IndexError | 已有保护（`base_objectnav_policy.py:159-162`）；A\* 输入 px 先 clip 到 `[0,size)` | — |

## 实施勾选清单

- [x] 新建 `vlfm/utils/path_planning.py`：`plan_path` / `snap_to_navigable` / `downsample_path` + (row,col)↔xy adapter。
- [x] 写 adapter 单测（xy→px→xy 往返一致；已知障碍布局下 A\* 绕行）。→ `test/test_path_planning.py`（8 项全过）
- [x] 新建 `vlfm/policy/utils/waypoint_controller.py`：`step_towards`（选项 B），turn_angle/arrive_radius 从常量/config 取。
- [x] `base_objectnav_policy.py`：加 `_navigate_global(goal_xy, conservative=False)` + 状态字段 + `_reset` 清理。
- [x] `base_objectnav_policy.py:act()`：navigate 分支按 `VLFM_GLOBAL_NAV` 切换；加 navigate-from-memory 支（接模块 2）。
- [ ] 验收 1（回家）→ 2（障碍上）→ 3（不连通）→ 4（回归）。 ← 需带 VLM 的 GPU 整跑

## 不在本模块范围

- 跨 start 重投影（Phase B，本期不做）。
- explore 的记忆偏置（可选增强，单列）。
- 选项 A 的 RNN lookahead 调优（默认选项 B，A 仅备选）。
