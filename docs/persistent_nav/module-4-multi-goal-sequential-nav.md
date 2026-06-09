# 模块 4:多目标顺序导航(NL 分解 + 已知/未知分流 + 回原位)

**改动量**:中　**依赖**:模块 1/2/3 的产物与运动原语(已落地)　**前置**:先做 Part A 简化　**开关**:`VLFM_GOAL_SEQUENCE`

## 目标

把 cat_demo 从「单目标一次性」升级为「一条自然语言里的**有序多目标**依次完成」:

> 原位 → 冰箱 → 猫 → 原位

每个目标按「记忆里是否已知」分流:

- **已知**(记忆 json 命中,如猫):A\* 离散点 + pointnav 连接,**直奔记忆点**,不浪费探索。
- **未知**(记忆没有,如冰箱):**原版 VLFM + nav 探索**(frontier + 价值图)找到为止,找到后顺手 `remember_object` 存下。
- **点目标**(原位 = episodic `(0,0)`):保守图 A\* 直接奔点。

复用模块 1/2/3 的**全部运动原语**;本模块新增的只有**排程层**(目标队列 + 换挡)与 **NL 分解**两件事。

---

## 核心洞察(决定了本模块为什么"小")

整套**感知 / 探索 / 检测 / 价值图 / 导航**栈**已经被 `self._target_object` 参数化**:

- 检测:`_get_object_detections` 按 `self._target_object.split("|")` 选 COCO/非 COCO 并过滤类别。
- 目标定位:`_get_target_object_location` 查 `self._object_map.has_object(self._target_object)`。
- 探索:价值图文本提示由 `self._target_object` 注入。
- 物体点云图 `ObjectPointCloudMap` **按类名 key 存**,多目标天然共存,**切目标只是查不同 key,不需要清空**。

**推论:多目标 = 用一个队列驱动 `self._target_object`(+ 点目标特例)+ 把"到点 STOP"从"整集结束"改成"换下一个目标"。** `act()` 现有的分支顺序(检测到→去 / 记忆里→A\* / 否则探索)对单目标恰好就是"已知走记忆、未知去探索"。

---

## Part A(前置改造):砍掉 geometric 跟随器与卡死/碰撞补丁

承接讨论结论:既然走 **A\* 离散点 + pointnav 连接**(pointnav 从实时 depth 自带局部避障),geometric 跟随器和它的卡死/碰撞补丁就是多余技术债,先删干净再做 Part B。

### 删除清单(`vlfm/policy/base_objectnav_policy.py`,除非另注)

- 常量:`_NAV_TURN_ANGLE_RAD`、`_NAV_STUCK_MOVE_EPS_M`、`_NAV_STUCK_STEPS`、`_NAV_COLLISION_MARK_RADIUS_M`、`_NAV_COLLISION_MARK_AHEAD_M`、`_NAV_STUCK_STOP_MARGIN_M`。
- 字段(`__init__`/`_reset`):`_last_nav_robot_xy`、`_last_nav_action_id`、`_nav_stuck_steps`。
- 方法:`_remember_nav_action`、`_update_nav_stuck_state`、`_mark_collision_ahead`、`_mark_current_waypoint_blocked`、`_mark_obstacle_disk`。
- `_navigate_global` 内:geometric 分支(`step_towards`)、`force_replan`/卡死逻辑、卡死-停车块(原 `:487-498`)、`_remember_nav_action` 包装。**改为只走 pointnav 跟随器。**
- import:`from vlfm.policy.utils.waypoint_controller import step_towards`。
- 整文件删除:`vlfm/policy/utils/waypoint_controller.py`。
- env:`VLFM_GLOBAL_NAV_FOLLOWER`、`VLFM_NAV_TURN_ANGLE_DEG` 的读取与文档。

### ⚠️ 一定保留(别误删)

- **`_needs_replan` 及其全部触发器**(目标漂移 `_NAV_GOAL_DRIFT_M`、每 `_NAV_REPLAN_PERIOD` 步、下一 waypoint 被新观测变 non-navigable)。这不是卡死补丁,而是「边走边建图、物体点云在动 → 旧 A\* 路径会过时」,**两种场景都需要**。
- 常量 `_NAV_ARRIVE_RADIUS`、`_NAV_WAYPOINT_SPACING_PX`、`_NAV_GOAL_DRIFT_M`、`_NAV_REPLAN_PERIOD`。
- `path_planning.py` 全部、`_plan_path_xy`、`conservative` 参数(回原位用保守图)。

### 简化后的 `_navigate_global`(目标形态,~30 行)

```python
def _navigate_global(self, goal_xy, conservative=False):
    robot_xy = self._observations_cache["robot_xy"]
    heading  = self._observations_cache["robot_heading"]
    rho_goal, _ = rho_theta(robot_xy, heading, goal_xy)
    if rho_goal < self._pointnav_stop_radius:
        self._called_stop = True
        return self._stop_action
    navigable = self._obstacle_map._navigable_map.astype(bool)
    if conservative:
        navigable = navigable & self._obstacle_map.explored_area.astype(bool)
    if self._needs_replan(goal_xy, navigable):
        path_xy = self._plan_path_xy(robot_xy, goal_xy, navigable)
        if path_xy is None:
            self._global_path = None
            return None                      # 真被围死 → 上层回落探索
        self._global_path, self._path_goal = path_xy, goal_xy
        self._waypoint_idx, self._last_plan_step = 0, self._num_steps
    while (self._waypoint_idx < len(self._global_path)
           and rho_theta(robot_xy, heading, self._global_path[self._waypoint_idx])[0] < _NAV_ARRIVE_RADIUS):
        self._waypoint_idx += 1
    if self._waypoint_idx >= len(self._global_path):
        self._called_stop = True
        return self._stop_action
    return self._pointnav(self._global_path[self._waypoint_idx], stop=False)
```

(`_pointnav` 自己管 `_last_goal`,目标 marker 仍正常显示;RNN 只在 waypoint 真正前进时 reset。)

---

## Part B(主体):多目标排程

### 新建 `vlfm/utils/goal_plan.py`(纯函数 + 数据类,可单测)

```python
@dataclass
class Goal:
    kind: str                  # "object" | "point"
    name: str = ""             # 物体名,如 "fridge"
    xy: Optional[np.ndarray] = None   # 点目标 xy,如 origin (0,0)

class GoalQueue:
    def __init__(self, goals: list[Goal]): ...
    def current(self) -> Optional[Goal]: ...
    def advance(self) -> bool:           # True=还有下一个;False=队列空
    @property
    def done(self) -> bool: ...

ORIGIN_WORDS = {"origin", "start", "原位", "起点", "出发点"}

def decompose(text: str) -> list[Goal]:
    # 受控自然语言 → 有序 Goal 列表:
    #  - 按 [>、,，。→ 和空白] 切分有序 token
    #  - origin 同义词 → Goal("point", xy=(0,0))
    #  - 其余 → Goal("object", name=token)
    # 预留 VLFM_GOAL_DECOMPOSER=llm 走 LLM 解析(本期默认规则解析)。

def resolve(goal: Goal, memory_path: str) -> str:
    # "known" | "unknown":点目标恒 known;物体看 recall_object 是否命中。
```

### 策略侧改造(`base_objectnav_policy.py`)

| 动作 | 落点 | 内容 |
|---|---|---|
| 新字段 | `__init__`/`_reset` | `self._goal_queue = None`、`self._multi_goal = False`、`self._memory_written = set()`(替代 `_memory_written_this_episode`) |
| 建队列 | `_pre_step`(`masks==0` 分支,设 `_target_object` 处附近) | `VLFM_GOAL_SEQUENCE` 有值 → `self._goal_queue = GoalQueue(decompose(...))`、`self._multi_goal=True`,并把首目标灌进 `self._target_object` |
| 目标来源改写 | `_pre_step` `:224` | 多目标时 **不**用 `observations["objectgoal"]`,改用 `self._goal_queue.current()` |
| 记忆按目标召回 | `_dispatch_goal` 内 | 物体目标 → `self._remembered_goal = recall_object(...)`;点目标 → 直接用 `goal.xy` |
| 换挡 | `act()` 尾部,`_num_steps += 1` 前 | 见下方「换挡逻辑」 |
| 每目标清状态 | 新 `_reset_per_goal_nav()` | 清 `_global_path/_path_goal/_waypoint_idx/_last_plan_step/_last_goal/_called_stop/_remembered_goal` + `_pointnav_policy.reset()` |

### `act()` 分支(多目标态)

点目标与物体目标统一走一个 `_dispatch_goal()`:

```python
def _dispatch_goal(self, observations):
    g = self._goal_queue.current()
    if g.kind == "point":
        return self._navigate_to(g.xy, observations, conservative=True)
    detected = self._get_target_object_location(self._observations_cache["robot_xy"])
    if detected is not None:                       # 看见了 → 直接去(已知/未知都适用)
        return self._navigate_to(detected[:2], observations, conservative=False)
    if self._remembered_goal is not None:          # 已知 → A* 直奔记忆点
        return self._navigate_to(self._remembered_goal, observations,
                                 conservative=True, fallback_to_pointnav=True)
    return self._explore(observations)             # 未知 → 原版 VLFM 探索
```

### 换挡逻辑(把"到点 STOP"改成"下一个目标")

`_navigate_to`/`_navigate_global` 到点仍照常置 `self._called_stop=True` 并返回 `_stop_action`(**单目标语义不变**)。多目标层在 `act()` 里**拦截**这一信号:

```python
if self._multi_goal and self._called_stop:
    g = self._goal_queue.current()
    if g.kind == "object":
        loc = self._get_target_object_location(robot_xy)
        if loc is not None and g.name not in self._memory_written and mem_path:
            remember_object(mem_path, g.name, loc[:2], ...)   # 未知物找到 → 记住
            self._memory_written.add(g.name)
    if self._goal_queue.advance():        # 还有下一个目标
        self._reset_per_goal_nav()         # 含清 _called_stop
        print(f"[goal] advance -> {self._goal_queue.current()}")
        pointnav_action = self._dispatch_goal(observations)   # 同步给出朝新目标的动作
    else:
        pass                                # 队列空:_called_stop 保持,真 STOP 终局
```

---

## 交付物 & 验收标准

### Part A(简化,可独立验)

- [ ] **A1**:`test/test_path_planning.py` 仍 **8/8 通过**(规划器未动)。
- [ ] **A2**:`grep` 确认 `_update_nav_stuck_state` / `_mark_collision_ahead` / `_mark_obstacle_disk` / `step_towards` 全部消失;`waypoint_controller.py` 已删;无 `VLFM_GLOBAL_NAV_FOLLOWER` / `VLFM_NAV_TURN_ANGLE_DEG` 残留。
- [ ] **A3(回归)**:不设 `VLFM_GLOBAL_NAV` 时,navigate 分支仍是原 `_pointnav(goal, stop=True)`,与现状一致。
- [ ] **A4(防误删)**:`_needs_replan` 及三触发器仍在;`_navigate_global` 仍按漂移/周期/waypoint 失效重算。
- [ ] **A5(GPU)**:`global_home_40` 预设仍能 A\*+pointnav 绕墙回原位并 STOP。

### Part B(多目标)

- [ ] **B1(单测)**:`decompose("原位,冰箱,猫,原位")` → `[point(0,0), object(fridge), object(cat), point(0,0)]`;另测 `>`/箭头/英文混排几种。
- [ ] **B2(单测)**:`resolve` —— 记忆有猫→`known`;记忆无冰箱→`unknown`;点目标→`known`。
- [ ] **B3(单测)**:`GoalQueue.current/advance/done` 语义;空列表保护。
- [ ] **B4(回归)**:不设 `VLFM_GOAL_SEQUENCE` 时行为与单目标数据集路径**逐字节一致**。
- [ ] **B5(端到端,GPU,两阶段)**:
  - **Pass 1**:原版 find-cat,开 `VLFM_PERSIST_MAP_PATH` + `VLFM_OBJECT_MEMORY_PATH` → 跑完 npz 存在、`cat.json` 含猫点(复用模块 1/2)。
  - **Pass 2**:`VLFM_GOAL_SEQUENCE="原位,冰箱,猫,原位"` + 载入同一 map/memory →
    - **原位**:开局即在,迅速 advance;
    - **冰箱(未知)**:进入 `explore`(frontier),检测到冰箱→navigate→advance;
    - **猫(已知)**:**不探索**,A\* 直奔记忆猫点→navigate→advance;
    - **原位**:保守图 A\* 回家→STOP。
  - 日志可见:`explore`→`navigate`(冰箱)、`navigate`/A\*(猫)、point-nav(原位)的模式切换 + `[goal] advance` 行;**仅最后一个原位到点才 STOP**。

> **跑 B5**:web 选 `multi_goal_cat`(两阶段一键);或 CLI 两步——
> ```bash
> # Pass 1:建图 + 记住猫(单目标 find-cat)
> VLFM_GLOBAL_NAV=1 VLFM_PERSIST_MAP_PATH=data/persistent_maps/cat.npz \
>   VLFM_OBJECT_MEMORY_PATH=data/object_memory/cat.json bash scripts/cat_demo/eval_cat_demo.sh
> # Pass 2:载入 + 多目标(原位→冰箱[探索]→猫[记忆 A*]→原位)
> VLFM_GLOBAL_NAV=1 VLFM_PERSIST_MAP_PATH=data/persistent_maps/cat.npz \
>   VLFM_OBJECT_MEMORY_PATH=data/object_memory/cat.json \
>   VLFM_GOAL_SEQUENCE="origin, refrigerator, cat, origin" bash scripts/cat_demo/eval_cat_demo.sh
> ```

---

## 坑与解法

**坑 ①(切目标不要清 object_map)。** `ObjectPointCloudMap` 按类名 key,切 `_target_object` 自动查新类;清空会丢掉「路过时已瞥见下一个目标」的免费先验。**只清每目标导航状态,不清 object_map / obstacle_map。**

**坑 ②(`_called_stop` 一信号两义)。** 它既是「到达当前目标」又是「整集终局」。多目标层在 `act()` 拦截:非末目标 → 清掉并换挡;末目标 → 放行终局。`_navigate_*` 内部语义保持不变,确保 B4 回归。

**坑 ③(记忆写一次/物体,非一次/集)。** 原 `_memory_written_this_episode` 是每集一次;多目标要每**物体**一次 → 换成 `self._memory_written: set[str]`,key 用目标名。

**坑 ④(换挡当步无新检测)。** 本步检测是按**旧**目标跑的;advance 后立即 `_dispatch_goal` 只能用记忆/探索给动作,新目标的**新鲜检测下一步才有**。可接受(换挡步通常给出转向/前进的合理动作)。

**坑 ⑤(点目标不需要检测)。** 原位无物体;`_dispatch_goal` 点目标分支直接保守图 A\* 奔 `(0,0)`,`rho<stop_radius` 即到。为省一次无意义检测,可在点目标时跳过 `_update_object_map`(可选优化,先不做也不报错)。

**坑 ⑥(初始化自旋一次/集)。** `_done_initializing` 仅集级 reset 清零,**不要**每目标重置,否则每换目标都原地转一圈。

**坑 ⑦(切目标 RNN 脏状态)。** 每目标 reset 调 `_pointnav_policy.reset()` 并清 `_last_goal`,避免上一段 pointnav 的隐藏态串味。

**坑 ⑧(episodic 帧前提)。** 记忆猫点是存盘那次的 episodic 帧;cat_demo 单一固定出生点 → 直接可用。换出生点须先补 Phase B 重投影(与模块 1/2 同前提)。

**坑 ⑨(未知物真找不到)。** `explore` 跑满步数仍没检测到冰箱 → 该目标无法完成。本期策略:维持探索直到 episode 步数耗尽(不强行跳过),日志告警;后续可加「探索预算超限则跳过该目标」。

**坑 ⑩(物体名要对齐检测器词表)。** 目标物体 token 经 `_get_object_detections` 按类名过滤,而检测器输出英文(MP3D/COCO)。→ 多目标里的物体名用**英文类名**(如 `refrigerator`/`cat`);`冰箱` 这类中文要等 `VLFM_GOAL_DECOMPOSER=llm` 把中文映射成英文类后才可用。`origin`/`原位` 是点目标,不走检测,不受此限。`multi_goal_cat` 预设默认 `"origin, refrigerator, cat, origin"`。

---

## 文件 / 函数级落点

| 文件 | 动作 |
|---|---|
| `vlfm/utils/goal_plan.py`(新) | `Goal`、`GoalQueue`、`decompose`、`resolve` |
| `test/test_goal_plan.py`(新) | B1/B2/B3 单测 |
| `vlfm/policy/base_objectnav_policy.py` | Part A 删除 + `_navigate_global` 简化;Part B 字段/`_pre_step` 目标源/`_dispatch_goal`/换挡/`_reset_per_goal_nav` |
| `vlfm/policy/utils/waypoint_controller.py` | **删除** |
| `scripts/cat_demo/web.py` 或 `eval_cat_demo.sh` | 新增 `multi_goal_cat` 预设(`VLFM_GOAL_SEQUENCE` + 载图/记忆),供 B5 |

## 实施勾选清单

- [ ] Part A:删卡死/碰撞/geometric,简化 `_navigate_global`,删 `waypoint_controller.py`,跑 A1–A4。
- [ ] `vlfm/utils/goal_plan.py`:`Goal`/`GoalQueue`/`decompose`/`resolve`。
- [ ] `test/test_goal_plan.py`:B1–B3,本地 `python test/...` 全过。
- [ ] 策略侧:字段 + `_pre_step` 目标源 + `_dispatch_goal` + 换挡 + `_reset_per_goal_nav`。
- [ ] 回归 B4(不设 `VLFM_GOAL_SEQUENCE`)。
- [ ] 预设 + GPU 整跑 B5(两阶段)。

## 不在本模块范围

- LLM 版 `decompose`(留 `VLFM_GOAL_DECOMPOSER=llm` 钩子,本期规则解析)。
- 探索预算 / 未知物找不到的跳过策略(坑 ⑨,后续)。
- 跨 start 世界系重投影(Phase B)。
- 非原位的命名地标点(原位以外的 point goal 库)。
