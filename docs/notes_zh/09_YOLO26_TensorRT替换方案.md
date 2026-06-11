# 09 · YOLO26 + TensorRT 替换 YOLOv7-e6e:思路 / 坑 / 验收

> 关键文件:
> - `vlfm/vlm/yolov7.py`(老 COCO 检测服务,默认权重 `data/yolov7-e6e.pt`,Flask 路由 `/yolov7`,默认端口 12184)
> - `vlfm/vlm/detections.py`(`ObjectDetections` 契约 = 新服务必须对齐的 JSON)
> - `vlfm/vlm/coco_classes.py`(标准 COCO-80;与 Ultralytics 类名/类序完全一致)
> - `vlfm/policy/base_objectnav_policy.py:84`(读环境变量 `YOLOV7_PORT` 实例化 `YOLOv7Client`)
> - `scripts/launch_vlm_servers_jy.sh:37`(tmux 起 4 个 VLM 服务,YOLO 这一行)
> - `scripts/cat_demo/eval_cat_demo.sh:68`(端口健康检查轮询)
>
> 目标:用 **YOLO26 + TensorRT** 替换 **YOLOv7-e6e** 作为 VLFM 的 COCO 检测器,提速,且不动 VLFM 主逻辑、不污染仿真环境。

---

## 9.0 背景与目标

- VLFM 的 COCO 检测器现在是 **YOLOv7-e6e**(老、慢、巨)。GroundingDINO 那条非 COCO 路**不在本次改动范围**。
- 换成 **YOLO26**(2026-01 发布,Ultralytics 最新,**NMS-free 端到端**,COCO-80;`yolo26n` ≈ 40.9 mAP / 2.4M 参数 / T4-TRT 1.7–11.8ms),用 **TensorRT engine** 推理。
- ONNX→TRT 不用手搓:Ultralytics 内置 `yolo export ... format=engine`。

---

## 9.1 为什么可行(三条架构事实)

1. **检测器是独立 Flask 进程,走 HTTP/JSON**(`vlfm/vlm/server_wrapper.py: host_model`)。策略侧只通过 `YOLOv7Client.predict(rgb) -> ObjectDetections` 调它。→ **换检测器不用动 VLFM 策略代码,也不要求与 VLFM 同 conda 环境。**
2. **切端口 = 一个环境变量。** `base_objectnav_policy.py:84` 读 `os.environ["YOLOV7_PORT"]`(默认 12184),`launch_vlm_servers_jy.sh:37` 也用它。
3. **JSON 契约极简且固定**(`detections.py: to_json`):
   ```json
   {"boxes": [[x1,y1,x2,y2], ...], "logits": [...], "phrases": ["chair", ...]}
   ```
   - `boxes`:**归一化 xyxy ∈ [0,1]**(客户端 `from_json` 用 `fmt="xyxy"` 还原)。
   - `phrases`:**COCO 类名字符串**。`coco_classes.py` 是标准 COCO-80,**与 Ultralytics `model.names` 完全一致**,所以是 near drop-in。

> 关键:新服务只要把 Flask 路由也注册成 `name="yolov7"`,**现有 `YOLOv7Client` 一行都不用改**,切端口即可生效。

---

## 9.2 工作流形态:三件事正交,别混为一谈

| 关注点 | 用什么解决 | 说明 |
|---|---|---|
| **代码回滚** | `git branch` | 在分支上改;不行就 `git checkout main`。✅ 取代"双端口灰度"那套仪式 |
| **端口** | 同一个 12184,直接 swap | 一次只跑一个 YOLO 服务;新服务路由 `name="yolov7"`,`YOLOv7Client` 不动 |
| **环境隔离** | 独立 `yolo_trt` conda env | **`git branch` 解决不了这个** —— 见下 |

⚠️ **核心提醒:`git branch` 回滚代码,回滚不了被污染的 conda 环境。** 往 `vlfm_cuda_sim`(锁死 `torch 2.1.0+cu121`)里 `pip install tensorrt ultralytics`,有把 torch/torchvision/numpy 顶替、连带搞坏 `habitat-sim` 的风险。因为检测服务是独立进程走 HTTP,**它跑在哪个 env 与端口/分支无关**,所以让它单独住在 `yolo_trt` env,代价几乎为零。

---

## 9.3 阶段与产物

### Phase 0 — 独立环境(不克隆旧环境)

机器:8×H20-3e(Hopper sm_90),驱动 570.172.08(支持到 CUDA 12.8)。

```bash
conda create -n yolo_trt python=3.11 -y
conda activate yolo_trt
pip install ultralytics              # 拉 torch(选 CUDA 12.x 轮子)
pip install tensorrt onnx onnxslim   # 导出 engine 需要
# 记录并 pin 三者版本:ultralytics / torch / tensorrt(导出环境必须 == 运行环境)
```
- 权重 `yolo26n.pt` 由 Ultralytics 自动从 GitHub/HF 拉,**走 7897 代理直接下**(见 docs 08 同源环境;代理已验证可达 github / raw / fbaipublicfiles)。
- **不要** `conda create --clone vlfm_cuda_sim`。

### Phase 1 — 独立跑通 TRT + 开源标准验收(不碰 VLFM)

```bash
# 1) 导出 engine(FP16)。必须在 H20 本机导;engine 绑 GPU 架构+TRT 版本,不可移植!
yolo export model=yolo26n.pt format=engine half=True imgsz=640 device=0
# 2) COCO val2017(Ultralytics 自动下,走代理)
yolo val model=yolo26n.engine data=coco.yaml imgsz=640 device=0   # 看 mAP
yolo val model=yolo26n.pt     data=coco.yaml imgsz=640 device=0   # 对照基线
# 3) 测速
yolo benchmark model=yolo26n.engine imgsz=640 device=0
```
**过 G1.* → commit。**

### Phase 2 — 接入(同端口 12184 swap)+ 三目标 A/B

新建 `vlfm/vlm/yolo_trt.py`(核心映射,务必对齐契约):

```python
# yolo_trt 环境内运行;不必依赖 vlfm 包,直接吐契约 dict 即可
from ultralytics import YOLO
import numpy as np, cv2
from vlfm.vlm.server_wrapper import ServerMixin, host_model, str_to_image  # 或自带等价实现

class YOLO26Trt:
    def __init__(self, engine="yolo26n.engine", imgsz=640, conf=0.25):
        self.model = YOLO(engine)      # NMS-free,内部已是端到端
        self.imgsz, self.conf = imgsz, conf
        # 启动断言:类序必须 == COCO-80,否则 fail-fast
        # assert list(self.model.names.values()) == COCO_CLASSES

    def predict(self, rgb: np.ndarray) -> dict:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)         # ⚠ Ultralytics numpy 输入按 BGR
        r = self.model(bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        H, W = rgb.shape[:2]
        xyxy = r.boxes.xyxy.cpu().numpy().astype(float)     # 像素
        xyxy[:, [0, 2]] /= W; xyxy[:, [1, 3]] /= H          # → 归一化
        xyxy = np.clip(xyxy, 0.0, 1.0)
        names = r.names
        return {
            "boxes":   xyxy.tolist(),
            "logits":  r.boxes.conf.cpu().numpy().tolist(),
            "phrases": [names[int(c)] for c in r.boxes.cls.cpu().numpy()],
        }

# 路由名必须是 "yolov7",这样现有 YOLOv7Client 不用改;端口仍 12184
# host_model(server, name="yolov7", port=12184)
```

启动脚本改 `scripts/launch_vlm_servers_jy.sh:37` 那一**行**:让 YOLO 这个 tmux pane **激活 `yolo_trt` env、跑新模块**(其余 3 个服务不动)。把 12184 加进 `eval_cat_demo.sh:68` 的健康检查轮询。

**三目标 A/B(收紧后的判据,不是"跑一次找得着"):**
- 目标:**cat / toilet / refrigerator(冰箱)** —— 三个都是 COCO 类,走的正是被替换的 YOLO 路径,选得对。
- **同 seed / 同场景**跟 `main` 跑**同一批 episode**,每个目标 **3–5 个**,不要单次。
- 判据 = **新分支成功的 episode ⊇ 原版成功的(无回退)**。
- 另加**检测级肉眼校验**:用 `ObjectDetections.annotated_frame` 挑 2–3 帧画框看,专抓 RGB/BGR、归一化、阈值这类胶水 bug(导航成败抓不到细微掉点)。

**过 G2.* → commit。**

### Phase 3 — merge

G1 + G2 都过 → 合回 `main`。**merge 时唯一别丢:把 `yolo_trt` env 配方 + 启动方式写进 repo**(conda/requirements 导出 + 改好的 launch 脚本),否则换机/队友拉下来跑不起来。老 `yolov7.py` 先留作 `.pt` fallback,稳定几轮后再清。

---

## 9.4 坑表(分组 + 对策)

| # | 坑 | 为什么咬你 | 对策 |
|---|---|---|---|
| 0 | **env 污染 ≠ 代码回滚** | `git branch` 回滚不了被 tensorrt/ultralytics 顶替的 torch,会拖垮 habitat-sim | 检测服务跑在独立 `yolo_trt` env(走 HTTP,本就不需同环境) |
| 1 | **engine 不可移植** | TRT engine 绑 GPU 架构(sm_90)+TRT 版本+CUDA;换机/换版本加载失败 | 在 H20 本机导出;pin 并记录 tensorrt 版本;导出环境 == 运行环境 |
| 2 | **RGB/BGR 反了** ⚠️ | VLFM 传 **RGB**;Ultralytics 对 numpy 输入默认按 **BGR** → 颜色错位、掉点 | predict 前 `cv2.cvtColor(RGB2BGR)`;靠 G2 检测框校验兜底(错了框会明显歪/少) |
| 3 | **坐标没归一化** | Ultralytics 返回**像素** xyxy;契约要 **[0,1]** | 除以 W/H 并 `clip(0,1)`(参照 `yolov7.py:100-103`) |
| 4 | **0.8 阈值套小模型漏检** | COCO 检测在策略侧用 `filter_by_conf(0.8)`(`docs 02` 2.2;给 e6e 调的);26n 置信分布不同 | 服务端只做低阈值粗筛(`conf≈0.25`),把 0.8 留给策略;必要时按 G2.2 重标定 `coco_threshold` |
| 5 | **类序/类集不一致** | 误用非 COCO 权重 → phrases 错位 | 用官方 COCO 权重;启动断言 `model.names == COCO_CLASSES`(标准权重两者一致,drop-in) |
| 6 | **重复 NMS** | YOLO26 已是 **NMS-free 端到端** | 直接用 Ultralytics 输出,别再叠 NMS |
| 7 | **imgsz 不匹配** | engine 按固定 imgsz 编译,运行时不一致报错/静默缩放 | 导出与推理统一 `imgsz=640` |
| 8 | **冷启动慢 / 同卡抢占** | 首推慢;新服务 + 其余 3 个 VLM 同卡 | 启动 warmup 数次;`CUDA_VISIBLE_DEVICES` 给新服务单独 pin 一张 H20(143GB,够) |
| 9 | **健康检查缺失** | eval 脚本不等端口就开跑 | 12184 纳入 `eval_cat_demo.sh:68` 轮询 |
| 10 | **success 指标对 cat/冰箱无效** ⚠️ | cat 是后加进场景的;冰箱是多目标定制(`VLFM_GOAL_SEQUENCE`),都不是 HM3D 原生目标类 | 这两项靠检测框/轨迹**肉眼判**;只有 **toilet**(HM3D 原生 6 类)success 指标可信 |

---

## 9.5 验收标准(量化闸门,过不了不进下一步)

**Phase 1(独立 TRT)**
- **G1.1** engine 在 H20 成功导出 + 加载 + warmup 通过。
- **G1.2** `coco.yaml` val:`.engine`(FP16)mAP 相对 `.pt` 掉点 **≤ 0.5**,且数量级吻合官方 `yolo26n`(~40.9 mAP@50-95)。
- **G1.3** 记录 ms/帧(640,bs=1,H20):engine 明显快于 `.pt`,远快于老 e6e(留数字做对照)。
- → **commit**

**Phase 2(接入)**
- **G2.1 检测级校验**:N≥? 帧(挑含目标类的若干帧),`annotated_frame` 目检框位置/类别正常,无系统性坐标/颜色错位。
- **G2.2 三目标 A/B**:`main` vs 新分支,**同 seed 同场景**,cat / toilet / refrigerator 各 3–5 episode;判据 = **新分支成功 episode ⊇ 原版**(toilet 看 success 指标;cat/冰箱看检测/轨迹)。
- **G2.3 阈值标定(条件触发)**:若 G2.2 因召回不足未过,标定新模型阈值使目标类召回 **≥ 老端口**。
- → **commit**

**Merge**
- G1.* + G2.* 全过;`yolo_trt` env 配方 + 启动脚本已入库;`.pt`/老 `yolov7.py` 保留作 fallback。

---

## 9.6 回滚

- **代码**:`git checkout main`(或 revert 分支)。
- **运行**:把 `launch_vlm_servers_jy.sh` 那行换回 `python -m vlfm.vlm.yolov7`(老 env、同端口 12184),或临时 `export YOLOV7_PORT=<老服务端口>`。
- **环境**:`yolo_trt` 与 `vlfm_cuda_sim` 互不影响,删 `yolo_trt` 即可。

---

## 9.7 待确认项

- [ ] **冰箱(refrigerator)是否走 success 指标**:确认它是否 HM3D 原生目标类。若是多目标定制目标(`VLFM_GOAL_SEQUENCE`),则同 cat 一样 success 无效,只能肉眼判。
- [ ] `coco_threshold` 的确切配置位置与默认值(策略侧 `filter_by_conf(0.8)`),便于 G2.3 标定。
- [ ] `yolo26n` 是否精度够用(室内 6+ 类);不够则升 `yolo26s/m` 再走同流程。
