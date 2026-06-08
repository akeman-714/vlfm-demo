# 模块 2：记忆 json（物体位置）

**改动量**：小　**依赖**：无（可选 `filelock`）　**前置模块**：无（消费方需模块 3 才有用）　**开关**：`VLFM_OBJECT_MEMORY_PATH`

## 目标

把「这一集在哪找到了目标物体」持久化成 json，下集开局直接读出 → 作为导航先验（先去记忆点，没找到再 explore）。
cat_demo 单 start → 存的 episodic 点下集直接可用。

## 交付物 & 验收标准

- [ ] `VLFM_OBJECT_MEMORY_PATH=data/object_memory/cat.json bash scripts/cat_demo/eval_cat_demo.sh` 跑一次。
- [ ] **验收 1**：跑完 json 存在，含 `{"cat": {"xy": [x, y], "start_pose": [...], "ts": ...}}`，`xy` 与视频里猫的位置吻合（量级对、不是 0,0）。
- [ ] **验收 2**：第二次跑，reset 后打印 `[memory] recalled cat at [x, y]`，且 `self._remembered_goal` 被正确填充。
- [ ] **验收 3（回归）**：不设 `VLFM_OBJECT_MEMORY_PATH` 时行为与现状一致。
- [ ] （端到端「直奔记忆点」效果在**模块 3** 联调时验收，本模块只保证**存得对、读得回**。）

## 数据源（现成，不用新算）

`get_best_object()` 返回 episodic 2D 点（`vlfm/mapping/object_point_cloud_map.py:77-100`）
→ `_get_target_object_location()`（`vlfm/policy/base_objectnav_policy.py:171-175`）
→ 即 `act()` 里的 `goal`（`vlfm/policy/base_objectnav_policy.py:128`）。
`get_best_object` 自带去抖（移动 <0.5m 不更新，`object_point_cloud_map.py:89-98`），**STOP 那刻取值最干净、置信最高**。

## 文件 / 函数级落点

| 动作 | 文件:行 | 内容 |
|---|---|---|
| 新建读写工具 | `vlfm/utils/object_memory.py`（新文件） | `remember_object` / `recall_object` |
| **写**：STOP 时存 | `vlfm/policy/base_objectnav_policy.py:act()` 尾部 | `goal is not None` 且 `_called_stop` → 存 `goal[:2]` |
| **读**：reset 后载入 | `vlfm/policy/base_objectnav_policy.py:_reset()` 末尾 | 填 `self._remembered_goal` |
| 新字段 | `__init__` / `_reset()` | `self._remembered_goal = None` |

### `vlfm/utils/object_memory.py`（建议签名）

```python
def remember_object(path, target, xy, start_pose=None):
    # 读改写一份 json：{target: {"xy": [..], "start_pose": [..], "ts": time()}}
    # 原子写：写 *.tmp -> os.replace；多 worker 时用 filelock 包住读改写
    ...

def recall_object(path, target):
    # 返回 np.array([x, y]) 或 None；target 用统一规范化 key（见坑⑥）
    ...
```

### 策略侧钩子

```python
# base_objectnav_policy.py:act() 尾部（navigate 段之后、num_steps 自增附近）
path = os.environ.get("VLFM_OBJECT_MEMORY_PATH")
if path and goal is not None and self._called_stop:
    remember_object(path, self._target_object, goal[:2], start_pose=...)

# base_objectnav_policy.py:_reset() 末尾
self._remembered_goal = None
path = os.environ.get("VLFM_OBJECT_MEMORY_PATH")
if path and self._target_object:  # 注意：_target_object 在 _pre_step 才被设置（见坑①）
    self._remembered_goal = recall_object(path, self._target_object)
```

## 坑与解法

**坑 ①（顺序）`_target_object` 在 `_reset()` 时还是空串。**
`_reset()` 里 `self._target_object = ""`（`base_objectnav_policy.py:95`），真正赋值在 `_pre_step` 的 `masks==0` 分支（`:156`，**在 `_reset` 之后**）。
→ 解法：**把 recall 移到 `_pre_step` 设置完 `_target_object` 之后**，或在 `act()` 首次进入时 lazy-load。不要在 `_reset` 内部 recall（那时 target 为空）。

**坑 ② in-memory 状态每集清空。** `object_point_cloud_map.py:25-27` 每 reset 清 `clouds`、`last_target_coord`。→ json 是唯一跨集载体，模块 3 消费 `self._remembered_goal`。

**坑 ③ 存哪一刻。** → STOP 时（`_called_stop`，由 `_pointnav:276` 或模块 3 控制器置位）。此刻 `get_best_object` 已收敛。

**坑 ④ 并发写。** 多 worker 同 target → `filelock` 包「读改写」；单 writer 用 `os.replace` 原子写即可。建议一 (scene,target) 一文件减少争用。

**坑 ⑤ 坐标帧 & 物体会动。**
存的是 episodic 点，仅同 start 可复用（与模块 1 同前提）；header 写 `start_pose` 作 Phase B 预留。
猫会动 → 记忆点可能过期。→ **当先验用**：模块 3 先去记忆点，到点没检测到目标就回落 explore，别当 ground truth。

**坑 ⑥ target 是 `|` 多类串**（mp3d 的 table 等：`"table|dining table|..."`）。
→ json key 统一规范化（建议取首段或整串，但读写一致）；cat_demo 是单一 `"cat"`，先按整串 key 即可。

## 实施勾选清单

- [ ] 新建 `vlfm/utils/object_memory.py`：`remember_object` / `recall_object`，原子写 + 可选 filelock。
- [ ] `base_objectnav_policy.py`：`__init__`/`_reset` 加 `self._remembered_goal=None`。
- [ ] recall 放到 `_pre_step`（target 设好之后），填 `self._remembered_goal`，打印 `[memory] recalled ...`。
- [ ] `act()` 尾部：STOP 且 `goal` 非空 → `remember_object`。
- [ ] 跑一次验收 1–3。

## 不在本模块范围

- 「直奔记忆点」的导航行为 = **模块 3** 的 navigate-from-memory 态（本模块只负责存/读数据）。
- 多物体/多场景的记忆库结构优化（先单 target 单文件）。
- 跨 start 重投影（Phase B，本期不做）。
