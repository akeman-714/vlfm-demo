# 持久化导航 — 总览与任务索引

把 cat_demo 从「每集从零探索」升级为「跨集复用障碍图 + 记忆物体位置 + A\* 全局奔点」。
三个模块**独立开发、独立交付**，本目录每个模块一份任务文档，便于逐块监督验收。

| 模块 | 文档 | 改动量 | 状态 |
|---|---|---|---|
| 1 持久化障碍图 | [module-1-persistent-obstacle-map.md](module-1-persistent-obstacle-map.md) | 小 | ✅ 已落地（f673463） |
| 2 记忆 json（物体点） | [module-2-object-memory.md](module-2-object-memory.md) | 小 | ✅ 已落地（f673463） |
| 3 全局导航（A\*+控制器） | [module-3-global-navigation.md](module-3-global-navigation.md) | 中 | ✅ 代码+单测已过；两阶段 web demo 跑通 |
| 4 多目标顺序导航 | [module-4-multi-goal-sequential-nav.md](module-4-multi-goal-sequential-nav.md) | 中 | ◑ 开发中（Part A 简化 + Part B 排程） |

> 交付顺序建议见文末「开发顺序」。每份文档是自包含的：目标、落点、勾选清单、坑与解法、验收标准。

---

## 0. 共享坐标锚点（已核实）

| 事实 | 出处 | 含义 |
|---|---|---|
| 出生点钉死在像素中心 `(500,500)`，`pixels_per_meter=20`，`size=1000` | `vlfm/mapping/base_map.py:23`、`vlfm/mapping/obstacle_map.py:32-33` | 地图覆盖出生点 ±25m |
| `_xy_to_px` / `_px_to_xy` 含 **y 轴翻转 + 行翻转** | `vlfm/mapping/base_map.py:35-60` | 所有 xy↔px **必须复用这两个**，禁止手写翻转 |
| 障碍数组索引为 `[row, col] = [px[:,1], px[:,0]]` | `vlfm/mapping/obstacle_map.py:101` | A\* 在数组里按 (row,col)，回转 xy 用 `_px_to_xy` |
| 越界已有保护 `IndexError → "Reached edge of map"` | `vlfm/policy/base_objectnav_policy.py:159-162` | 大场景超 ±25m 触发；cat_demo 不会 |

## 0.1 cat_demo 前提（已核实，决定可以跳过 Phase B）

- `data/datasets/objectnav/hm3d/v1/cat_demo/content/TEEsavR23oF.json.gz` 只有 **1 个 episode**，
  `start_position=[-8.433, 0.163, -2.377]`、`start_rotation=[0.0, 0.858, 0.0, 0.513]` **固定**。
- 每次 Run 都用同一 start → **episodic 帧恒等、零变换**。持久化图与记忆点直接对齐，无需重投影。
- ✅ **本期不做 Phase B**（跨 start 的世界坐标注入）。三模块只在 episodic 帧（相对出生点）工作。
  各模块仍会把 `start_pose` 写进存档 header，作为将来 Phase B 的预留，但代码路径不依赖它。

## 0.2 ⚠️ 贯穿三模块的运行时事实：每次 Run = 独立子进程 + 单 episode

- web 入口 `scripts/cat_demo/web.py` 的 `_run_eval` **每次点 Run 都 spawn 新的 `python -m vlfm.run`**；
  `scripts/cat_demo/eval_cat_demo.sh` 里 `N_EP=1`。
- 推论 A：**跨集复用 = 跨进程复用 = 只能走文件**（内存状态每进程销毁）。
- 推论 B：`_reset()` 在 N_EP=1 时**只触发一次**，没有「下一次 reset 顺手存上一集图」的机会。
  → **模块 1/2 的「写盘」必须发生在 episode 进行中（周期性）或 STOP 时，不能挂在 next-reset。**

---

## 包选型决策（可引入新包）

| 用途 | 选用 | 理由 | 备选/升级 |
|---|---|---|---|
| 网格最短路（A\*） | `skimage.graph.route_through_array`（geometric=True） | 已装（skimage 0.24.0），实测可用、对角感知、零新依赖 | `pyastar2d`（C 加速真 A\*，启发式不展开全图）—— 若 1000² 重规划成性能瓶颈再换，接口同为 (start_px, goal_px) |
| 目标 snap 到最近可走像素 | `scipy.ndimage.distance_transform_edt(return_indices=True)` | 已装（scipy 1.13.1），一次拿到最近可走点 | — |
| json 并发写锁 | `os.replace` 原子写（基线，零依赖）；多 worker 再加 `filelock` | 单 writer 用原子写足够 | `pip install filelock` |

> 原则：cat_demo 现有环境（cv2 4.5.5 / numpy 1.26.4 / scipy / skimage / networkx 都在 `vlfm_pip`）已够用，
> 默认**不新增依赖**；仅当 `pyastar2d`/`filelock` 被证明必要时再装，并在对应模块文档登记。

---

## 非侵入开关（便于 A/B 与监督）

三模块都以**环境变量 opt-in**，不设变量时行为与现状完全一致：

| 开关 | 作用 | 默认 |
|---|---|---|
| `VLFM_PERSIST_MAP_PATH` | 设置后启用模块 1：障碍图存/读到该 npz 路径 | 未设=关闭 |
| `VLFM_OBJECT_MEMORY_PATH` | 设置后启用模块 2：物体记忆 json 路径 | 未设=关闭 |
| `VLFM_GLOBAL_NAV` | `=1` 启用模块 3：navigate 分支走 A\*+控制器；否则保持原 `_pointnav` | 未设=关闭 |

`scripts/cat_demo/eval_cat_demo.sh` 里 export 这些变量即可逐模块开启，方便单独验收。

---

## 开发顺序（先暴露最难的模块 3）

1. **模块 3**：在 live 图上，目标先设成回出生点 `(0,0)`，跑通「绕墙回家 + STOP」。不依赖 1/2。
2. **模块 1**：存/merge **仅障碍 `_map`**。跑两次，第二次开局图已预填。
3. **模块 2**：json 存猫点 → 把目标从 `(0,0)` 换成记忆猫点。
4. （Phase B 跨 start —— 本期不做。）

## 统一验收入口

```bash
# 单跑一集，产出视频到 video_dir/cat_demo_<时间戳>/
bash scripts/cat_demo/eval_cat_demo.sh
# 或 web： bash scripts/cat_demo/web.sh
```
每个模块文档末尾给出该模块**具体**的观察点与通过标准。
