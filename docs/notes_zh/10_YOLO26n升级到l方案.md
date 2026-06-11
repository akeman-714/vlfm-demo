# 10 · YOLO26n → YOLO26l 升级:指导方案 / 坑 / 验收

> 前置:本文是 [09_YOLO26_TensorRT替换方案.md](./09_YOLO26_TensorRT替换方案.md) 的后续。09 已把 YOLOv7-e6e 换成 YOLO26n+TensorRT 并跑通;本文只解决「把检测精度更高的 **yolo26l** 换上去」这一件事,**复用 09 建立的全部架构、环境与闸门语义**。
>
> 关键文件(均已就位,本次基本只改参数):
> - `vlfm/vlm/yolo_trt.py`(TRT 检测服务,路由 `/yolov7`,默认 `data/yolo26n.engine`)
> - `scripts/yolo_trt/export_engine.sh`(导出 engine;已用 `YOLO_TRT_WEIGHTS` 参数化)
> - `scripts/yolo_trt/setup_env.sh` + `requirements-pinned.txt`(`yolo_trt` env 配方,**升级不动它**)
> - `scripts/launch_vlm_servers_jy.sh:16`(读 `YOLO_TRT_MODEL`,默认 `data/yolo26n.engine`)
> - `scripts/cat_demo/eval_yolo26_targets.sh`(cat/toilet/冰箱 A/B)
> - `vlfm/policy/base_objectnav_policy.py:77`(`coco_threshold=0.8`,COCO 路置信阈值)
>
> 目标:用 **yolo26l** 提升室内 COCO 类(cat / toilet / refrigerator / chair / couch …)的**召回**,不碰 VLFM 主逻辑、不动 conda 环境、不换端口。

---

## 10.0 背景与动机

- 09 落地的是 **yolo26n**(nano,≈40.9 mAP / 2.4M 参数)。09 的 [9.7 待确认项] 已明确预留升级口:「`yolo26n` 是否精度够用;不够则升 `yolo26s/m`(同理 `l`)再走同流程」。
- 实跑后若发现室内目标类**漏检 / 召回不足**(尤其在策略侧 `0.8` 阈值下被砍掉),换更大的 `yolo26l` 是对症之举。
- **2026-06-11 H20 实测**(coco_val2017 5000 张,bs=1,imgsz=640,FP16 engine):

  | | 参数量 / GFLOPs | COCO val2017 mAP50-95 | H20 TRT-FP16 推理 ms/帧 |
  |---|---|---|---|
  | yolo26n(现状)| ≈2.4M | **39.7**(实测,FP16 engine)| **1.5** |
  | **yolo26l(本次)** | **24.8M / 86.4G**(实测)| **53.6**(实测 FP16 engine;.pt 53.8 → 掉点 0.2)| **2.8** |

  → **+13.9 mAP(n/l 均 FP16 engine、同盘同数据),延迟仅 +1.3ms/帧、绝对 2.8ms** —— 远非 nav loop 瓶颈(GDINO/BLIP 更重)。一次「拿 1.3ms 换 +13.9 mAP 召回」的低风险交易,**G1.\* 已全过(见 10.5)**。

---

## 10.1 为什么低风险(三条架构事实,全部来自 09)

1. **检测器是独立 Flask 进程,走 HTTP/JSON。** 策略侧只通过 `YOLOv7Client.predict(rgb)` 调它 → 换模型不动 VLFM 策略,也不要求同 conda 环境。
2. **模型路径已全程参数化。** `yolo_trt.py` 收 `--model`;`launch_vlm_servers_jy.sh` 读 `YOLO_TRT_MODEL`;`export_engine.sh` 读 `YOLO_TRT_WEIGHTS`。**n→l 本质 = 重导一个 engine + 改一个环境变量。**
3. **l 仍是 COCO-80,类名/类序与 `coco_classes.py` 完全一致。** `yolo_trt.py:51` 的 `class_names == COCO_CLASSES` 断言对 l 同样成立 → 依旧 drop-in,`phrases` 不会错位。

> 而且 **`yolo_trt` env 完全复用**:l 的导出/运行用的是同一套 `ultralytics 8.4.63 / torch 2.11.0+cu128 / tensorrt-cu12 10.9.0.34`(`requirements-pinned.txt`)。09 里最重的「独立环境 + pin 版本」工作量,本次为零。

---

## 10.2 改动面(极小)

| 改什么 | 怎么改 | 是否必须 |
|---|---|---|
| **导出 l engine** | `YOLO_TRT_WEIGHTS=yolo26l.pt bash scripts/yolo_trt/export_engine.sh` → 产出 `data/yolo26l.engine` | ✅ 必须 |
| **指向新 engine** | 起服务时 `YOLO_TRT_MODEL=data/yolo26l.engine bash scripts/launch_vlm_servers_jy.sh` | ✅ 必须 |
| 改默认值(可选清理)| 把下列写死 `yolo26n` 的点改成可切 `n/l`,见下方清单 | ⬜ cosmetic |
| imgsz | **保持 640**,导出与推理统一(launch 不传 `--imgsz`,server 默认 640,export 默认 640)| ✅ 别动 |

**写死 `yolo26n` 的位置清单(处理情况):**
- ✅ `scripts/yolo_trt/export_engine.sh`(2026-06-11 已参数化):新增 `ENGINE="${MODEL%.pt}.engine"`,engine 名随 `YOLO_TRT_WEIGHTS` 派生;echo / 可选 val·benchmark 行不再写死 `yolo26n.engine`。跑 `YOLO_TRT_WEIGHTS=yolo26l.pt bash …/export_engine.sh` 即出 `data/yolo26l.engine`。
- `vlfm/vlm/yolo_trt.py:32/:103`、`scripts/launch_vlm_servers_jy.sh:16`:默认值仍 `yolo26n`,但本就可经 `--model` / `YOLO_TRT_MODEL` env 覆盖 → **已可切 n/l**,默认刻意留 n 作 fallback,Phase 2 验收稳定后再翻(见 10.3 Phase 3)。

> 建议:**保留 `data/yolo26n.engine` 不删**,作为「掉速/出问题」的即时 fallback(回滚 = 把 `YOLO_TRT_MODEL` 换回去,零成本)。

---

## 10.3 阶段与产物(复用 09 的 G1/G2 闸门)

### Phase 1 — 独立导出 + 开源标准验收(不碰 VLFM)

```bash
# 在 H20 本机、yolo_trt env 内(engine 绑 sm_90 + TRT 版本,不可移植 → 必须本机导)
# 1) 导出 l engine(自动下 yolo26l.pt,走 7897 代理)
YOLO_TRT_WEIGHTS=yolo26l.pt bash scripts/yolo_trt/export_engine.sh
#    等价手命令:
#    cd data && yolo export model=yolo26l.pt format=engine half=True imgsz=640 device=0

# 2) 精度对照(COCO val2017)。⚠️ 别直接 data=coco.yaml:它会拉全量 ~20GB(train+val+test)
#    且 check_det_dataset 强制要 `train:` 键(见坑 L6)。只下 val2017 + 写 val-only yaml:
#      labels: ${ASSETS_URL}/coco2017labels-segments.zip   images: val2017.zip  → /data/datasets/coco
#      coco_val.yaml: path=/data/datasets/coco ; train: 与 val: 都指 val2017.txt ; 80 names
DATA=/data/datasets/coco_val.yaml
yolo val model=yolo26l.engine data=$DATA imgsz=640 device=0   # FP16 engine mAP  (实测 53.6)
yolo val model=yolo26l.pt     data=$DATA imgsz=640 device=0   # .pt 基线          (实测 53.8 → 掉点 0.2)
yolo val model=yolo26n.engine data=$DATA imgsz=640 device=0   # n 对照,同盘同数据 (实测 39.7)

# 3) 测速:val 输出自带 "Speed: …ms inference per image" 即 G1.3(本次 l=2.8ms / n=1.5ms),
#    无需单跑 benchmark(它会把模型重导成多种格式,反而绕)。
```

**过 G1.* → commit。**

### Phase 2 — 接入(同端口 12184 swap)+ 三目标 A/B

```bash
# 起服务时指向 l engine,其余 3 个 VLM 服务不动
YOLO_TRT_MODEL=data/yolo26l.engine bash scripts/launch_vlm_servers_jy.sh
# 等 60–90s 权重 + warmup,再跑 A/B
bash scripts/cat_demo/eval_yolo26_targets.sh
```

**三目标 A/B(同 09 判据,不是「跑一次找得着」):**
- 目标:**cat / toilet / refrigerator**,各 **3–5 个 episode**,**同 seed / 同场景**。
- 判据 = **l 成功的 episode ⊇ n 成功的(无回退)**;只有 **toilet**(HM3D 原生类)看 success 指标,cat/冰箱看检测框/轨迹肉眼判(见 09 pit #10)。
- **重点盯阈值**:在 `coco_threshold=0.8` 下,l 的召回是否 ≥ n;不够则按 G2.3 重标(见坑表 L3)。

**过 G2.* → commit。**

### Phase 3 — 收尾

- l 表现稳定后,可把 `YOLO_TRT_MODEL` / 各处默认值改成 `yolo26l.engine`,并在本文勾掉验收项。
- `data/yolo26n.engine` 与老 `yolov7.py` 继续保留作 fallback,稳定数轮再清。

---

## 10.4 坑表

### A. n→l 专属(本次新增 / 放大,优先看这组)

| # | 坑 | 为什么 n→l 才咬你 | 对策 |
|---|---|---|---|
| **L1** | **`yolo26l` 权重可用性 / 命名未验证** ⚠️ | YOLO26 系 2026-01 发布,边界期;n 已确认,l 是否随官方放出、字符串是否就是 `yolo26l` 需确认 | 先跑 Phase 1 step 1:`yolo export` 第一步会自动下权重,**下不到即此 variant 未放出** → 退 `yolo26m`;别盲目继续 |
| **L2** | **延迟上一个台阶** | l 参数 ≈10×n(2.4M→~20–25M),T4-TRT 推断 ~2ms→~6–8ms;检测器每帧被 nav loop 调 | H20 上绝对值仍个位数 ms,大概率非瓶颈(GDINO/BLIP 更重);**但必须 G1.3 实测留数**;若进 critical path 再议(降 imgsz / 单卡 pin) |
| **L3** | **`0.8` 阈值需重新标定** ⚠️ | 策略侧 `coco_threshold=0.8`(`base_objectnav_policy.py:77`)是给 e6e 调的,n 已分布不同,l 置信分布**又**变(大模型正检置信通常更高/更集中) | 服务端继续 `conf=0.25` 粗筛不动;**G2.2 A/B 必跑**,大概率触发 G2.3:标定使目标类召回 ≥ n |
| **L4** | **导出更慢 / engine 更大 / 构建期峰值显存更高** | TRT 对大模型做 tactic search,耗时几分钟;engine 8MB→~50MB;编译瞬时占显存更多 | 一次性成本,H20 143GB 无压力;别在导出同时把卡占满 |
| **L5** | **FP16 掉点对大模型一般更小,但仍要测** | l 层多、数值范围不同;FP16 量化误差通常 ≤0.5 mAP,但不能默认 | G1.2 用 `.engine` vs `.pt` 对照,掉点 ≤0.5 才算过(**实测仅 0.2 ✅**)|
| **L6** ✅已踩 | **G1.2 数据集陷阱:`coco.yaml` 拉全量 ~20GB + val-only yaml 必须含 `train:` 键** | 跑 val 才咬:`coco.yaml` 的 download 段拉 train+val+test(~20GB);且 `check_det_dataset` 强制要 `train:`,缺了直接 `SyntaxError: 'train:' key missing` | 只下 `val2017.zip` + `coco2017labels-segments.zip` 到 `/data/datasets/coco`,写 val-only yaml,`train:`/`val:` 同指 `val2017.txt`(val 模式不读 train → 零额外下载)|

### B. 仍然适用的老坑(来自 09,不变,提醒勿忘)

| # | 坑 | 对策(本次复用) |
|---|---|---|
| pit #1 | **engine 不可移植**(绑 sm_90 + TRT 10.9 + CUDA)| 在 H20 本机导;复用同一 `yolo_trt` env,**无新环境工作量** |
| pit #2 | RGB/BGR | `yolo_trt.py:75` 已 `cvtColor(RGB2BGR)`,不动 |
| pit #3 | 坐标归一化 | `yolo_trt.py:81-84` 已除 W/H + clip,不动 |
| pit #6 | 重复 NMS | YOLO26 端到端 NMS-free,l 同样,直接用输出 |
| pit #7 | imgsz 不匹配 | 导出 / 推理统一 640,**别只改一头** |
| pit #8 | 冷启动 / 同卡抢占 | `yolo_trt.py:58-61` 已 warmup 3 次;l 更重 → 更有理由单卡 pin `CUDA_VISIBLE_DEVICES` |
| pit #9 | 健康检查 | 12184 已在 `eval_cat_demo.sh` 轮询内 |
| pit #10 | success 指标对 cat/冰箱无效 | cat/冰箱靠检测框/轨迹肉眼判,只 toilet 看 success |
| (09 G2 环境) | localhost 代理陷阱 | A/B 脚本已 `export no_proxy=127.0.0.1,localhost`,勿删 |

---

## 10.5 验收标准(量化闸门,过不了不进下一步)

**Phase 1(独立 TRT)— ✅ 全过 @2026-06-11(H20,yolo_trt env,device=1)**
- **G1.1 ✅** `yolo26l.pt`(50.7MB,`assets/releases/v8.4.0`,COCO-80 顺序确认)下载成功;`yolo26l.engine`(51.9MB)导出成功(ONNX 95MB → TRT FP16,构建 440s)、val 正常加载。
- **G1.2 ✅** coco_val2017(5000 张):`.engine` FP16 mAP50-95 = **53.6**,`.pt` 基线 **53.8** → **掉点 0.2 ≤ 0.5**;且 **远高于 n 的 39.7(同盘同数据 FP16 engine,+13.9 mAP)**。
- **G1.3 ✅** 推理 ms/帧(640,bs=1,H20):l engine **2.8ms**(总 ~3.5ms 含 pre/post),n engine 1.5ms;l 仅 +1.3ms、个位数 → 非 nav loop 瓶颈。
- → **commit**(本次)

**Phase 2(接入)**
- **G2.1 检测级校验**:挑若干含目标类帧,`annotated_frame` 框位置/类别正常,无系统性坐标/颜色错位(沿用 `scripts/yolo_trt/smoke_test.py`)。
- **G2.2 三目标 A/B**:`yolo26n` vs `yolo26l`,同 seed 同场景,cat/toilet/refrigerator 各 3–5 episode;判据 = **l 成功 episode ⊇ n**(toilet 看 success;cat/冰箱看检测/轨迹)。
- **G2.3 阈值标定(条件触发)**:若 G2.2 因召回/误检变化未过,标定 `coco_threshold` 使目标类召回 **≥ n**、误检不显著上升。
- → **commit**

**收尾**
- G1.* + G2.* 全过;`data/yolo26l.engine` 入位;`data/yolo26n.engine` + 老 `yolov7.py` 保留作 fallback;各处默认值(可选)切到 l。

---

## 10.6 回滚

- **模型**:`YOLO_TRT_MODEL=data/yolo26n.engine` 重起 launch(零成本,engine 还在)。
- **代码**:`git checkout main` 或 revert 本次提交。
- **运行(彻底退回 YOLOv7)**:把 `launch_vlm_servers_jy.sh` 的 0.3 pane 换回 `${prefix} + python -m vlfm.vlm.yolov7`(见该脚本头部回滚注释)。
- **环境**:`yolo_trt` env 不动(n/l 共用),无需重建。

---

## 10.7 待确认项

- [x] **G1.1**:`yolo26l.pt` 已放出(`assets/v8.4.0`),命名即 `yolo26l`,COCO-80 顺序确认 → 不退 `yolo26m`。
- [x] **G1.2**:实测 l engine 53.6 / .pt 53.8(掉点 0.2);n engine 39.7(+13.9);已填回 10.0 表。
- [x] **G1.3**:l engine 推理 2.8ms/帧(H20,640,bs=1)vs n 1.5ms → 非瓶颈。
- [ ] **G2.3**(Phase 2 待跑):`0.8` 阈值在 l 下是否需重标;重标后的 `coco_threshold` 值。
