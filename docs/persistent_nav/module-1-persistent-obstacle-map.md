# 模块 1：持久化障碍图

**改动量**：小　**依赖**：无（仅 numpy）　**前置模块**：无　**开关**：`VLFM_PERSIST_MAP_PATH`

## 目标

让障碍图跨 episode（跨进程）累积：每集把当前障碍栅格存盘，下集开局 OR 回内存，新观测继续往上叠。
cat_demo 单 start → episodic 帧恒等，**零变换对齐**。

## 交付物 & 验收标准

- [ ] `VLFM_PERSIST_MAP_PATH=data/persistent_maps/cat.npz bash scripts/cat_demo/eval_cat_demo.sh` 连跑两次。
- [ ] **验收 1**：第一次跑完，该 npz 存在且 `_map.sum() > 0`。
- [ ] **验收 2**：第二次跑的 **step 0**，`self._obstacle_map._map.sum() > 0`（开局即带障碍）。打印一行 `[persist] loaded N obstacle px` 佐证。
- [ ] **验收 3**：第二次跑的 `obstacle_map.visualize()`（视频左图/`policy_info["obstacle_map"]`）开局就有黑色障碍，而非全白。
- [ ] **验收 4（回归）**：不设 `VLFM_PERSIST_MAP_PATH` 时，行为与现状完全一致（无存盘、无读盘）。

## 设计要点（只存一张图）

- **只存 `_map`（障碍 bool）** + header `{size, pixels_per_meter, start_pose}`。
- **不存 `_navigable_map`**：派生量，load 后重算。
- **不存 `explored_area`**：见坑 ②（每帧被覆盖，存了也会被冲掉）。
- 存储：`np.savez_compressed`（1000² bool 压缩后极小）。**原子写**：`np.savez` 到 `*.tmp` → `os.replace`。

## 文件 / 函数级落点

| 动作 | 文件:行 | 内容 |
|---|---|---|
| 加 `save()` / `load_and_merge()` | `vlfm/mapping/obstacle_map.py`（类 `ObstacleMap`） | 见下方签名 |
| 抽 `_recompute_navigable()` | `vlfm/mapping/obstacle_map.py`，从 `update_map():105-109` 抽出 | merge 后调用 |
| **读**：reset 末尾 merge | `vlfm/policy/base_objectnav_policy.py:_reset()`，在 `self._obstacle_map.reset()`（`:103`）**之后** | 新增 `self._load_persistent_obstacles()` |
| **写**：act 尾部周期存 | `vlfm/policy/base_objectnav_policy.py:act()`，`self._num_steps += 1`（`:145`）前后 | 周期 + STOP 时 |

### `ObstacleMap` 新方法（建议签名）

```python
def _recompute_navigable(self) -> None:
    # 从 update_map:105-109 抽出，merge 后/需要时复用
    self._navigable_map = 1 - cv2.dilate(
        self._map.astype(np.uint8), self._navigable_kernel, iterations=1
    ).astype(bool)

def save(self, path: str, start_pose=None) -> None:
    # 原子写：tmp -> os.replace；只存 _map + header
    ...

def load_and_merge(self, path: str) -> int:
    # 读 npz；校验 size/pixels_per_meter 一致；self._map |= loaded_map
    # 调 self._recompute_navigable()；返回合并进来的障碍像素数
    ...
```

### 策略侧钩子

```python
# base_objectnav_policy.py:_reset() 末尾（compute_frontiers 为真时）
def _load_persistent_obstacles(self):
    path = os.environ.get("VLFM_PERSIST_MAP_PATH")
    if path and self._compute_frontiers and os.path.exists(path):
        n = self._obstacle_map.load_and_merge(path)
        print(f"[persist] loaded {n} obstacle px from {path}")

# base_objectnav_policy.py:act() 尾部
def _maybe_save_obstacles(self):
    path = os.environ.get("VLFM_PERSIST_MAP_PATH")
    if path and self._compute_frontiers and (
        self._num_steps % 10 == 0 or self._called_stop
    ):
        self._obstacle_map.save(path, start_pose=...)  # start_pose 见坑①
```

## 坑与解法

**坑 ① 帧对齐只在「同 start」成立。**
gps/compass 相对 *episode 起点*。cat_demo 单 start → 零变换，OR 即对齐。
→ 解法：header 顺带写 `start_pose`（起始 yaw 取 `_observations_cache["habitat_start_yaw"]`，见 `habitat_policies.py:248`），仅作 Phase B 预留；**本期 load 时不依赖它**。多场景共用一文件会串味 → 用 `VLFM_PERSIST_MAP_PATH` 让启动脚本按场景指定路径。

**坑 ②（关键）不要持久化 `explored_area`。**
`obstacle_map.py:146` 是 `self.explored_area = new_area`（**重新赋值**，只保留含 agent 的连通域，`:128-146`）。你 OR 进去的历史探索区，只要和当前 agent 不连通，**下一帧就被删**。
→ 解法：**只持久化单调的 `_map`**。`_map` 全程只 `=1`（`:101`），只有 `reset()` 清零（`:48-53`），OR 进去稳定不被冲。explored/frontier 重新累积，便宜且安全。

**坑 ③ merge 后 navigable 暂时是旧的。**
`_navigable_map` 只在 `update_map(update_obstacles=True)` 内重算（`:105`）。
→ 自愈：episode 起点顺序为 `_reset()`(load+merge) → `_pre_step` → `_cache_observations` → `update_map`（`habitat_policies.py:205`），**首帧 update_map 就用合并后的 `_map` 重算 navigable**。为 step-0 正确，merge 后仍显式调一次 `_recompute_navigable()` 兜底。

**坑 ④ 并行写同一文件。**
git 历史有「1+n 并行跑 episode」。
→ 原子 `os.replace` 必做；多 worker 同场景再上 `filelock`。`VLFM_PERSIST_MAP_PATH` 按 worker/场景分文件可彻底回避。

**坑 ⑤ 尺寸漂移。** load 时 assert `size`、`pixels_per_meter` 与当前一致，不一致**跳过 merge 并告警**（不要静默错位）。

## 实施勾选清单

- [ ] `obstacle_map.py`：抽 `_recompute_navigable()`，并让 `update_map():105-109` 改调它（去重）。
- [ ] `obstacle_map.py`：实现 `save()`（原子写、只存 `_map`+header）。
- [ ] `obstacle_map.py`：实现 `load_and_merge()`（校验 header、OR `_map`、重算 navigable、返回计数）。
- [ ] `base_objectnav_policy.py:_reset()`：末尾调 `_load_persistent_obstacles()`（env 开关 + compute_frontiers 守卫）。
- [ ] `base_objectnav_policy.py:act()`：尾部调 `_maybe_save_obstacles()`（周期 + STOP）。
- [ ] 跑两次验收 1–4。

## 不在本模块范围

- explored_area / frontier 的持久化（明确不做，见坑②）。
- 跨 start 重投影（Phase B，本期不做）。
- 多场景自动按 scene_id 分文件（暂由 `VLFM_PERSIST_MAP_PATH` 手动指定）。
