# 11 · SigLIP2-ITM 可回滚替换 BLIP2-ITM:规划 / 坑 / 解决方案 / 验收

> 关键文件:
> - `vlfm/vlm/blip2itm.py`(当前 BLIP2-ITM 服务与 `BLIP2ITMClient`,Flask 路由 `/blip2itm`,默认端口 12182)
> - `vlfm/policy/itm_policy.py:48`(`BaseITMPolicy` 只通过 `BLIP2ITMClient(port=BLIP2ITM_PORT)` 调 `cosine(image, text) -> float`)
> - `vlfm/mapping/value_map.py`(把 ITM cosine 投到 value map;主要用于 frontier 相对排序)
> - `config/experiments/reality.yaml:10`(真机配置使用 `RealityITMPolicyV2`)
> - `scripts/launch_vlm_servers_jy.sh:44`(tmux 启动 BLIP2-ITM pane)
> - `vlfm/vlm/server_wrapper.py`(HTTP/JSON 编解码与服务托管工具)
>
> 目标:新增一个 **SigLIP2-ITM 后端**,用更小/更省显存的双塔图文相似度模型替换 12182 后面的服务,policy / value map / reality 配置不动。2026-06-14 cat demo 单集成功后,SigLIP2-base 已批准作为默认 ITM 后端;效果不好时仍可一条启动命令回滚 BLIP2。

---

## 11.0 结论先行

最终落点是 **旁路新增 + 显式开关 + 默认 SigLIP2-base + 可回滚 BLIP2**。

核心原因:

1. 导航代码并不直接依赖 BLIP2 类,只依赖 `BLIP2ITMClient.cosine(image, text) -> float`。
2. client 发往 `http://localhost:${BLIP2ITM_PORT}/blip2itm`,服务返回 `{"response": float}` 即可。
3. `RealityITMPolicyV2` 的 frontier 选择是对 value map 做相对排序,不是用 BLIP2 绝对 cosine 阈值 gate 决策。
4. 因此新服务只要兼容 **同端口 / 同路由 / 同返回格式**,下游无需知道它后面是 BLIP2 还是 SigLIP2。

推荐启动形态:

```bash
# 默认:SigLIP2-base ITM(独立 siglip2_itm env,本机优先用 /data/jinsong.yuan/siglip2-base-patch16-384)
bash scripts/launch_vlm_servers_jy.sh 0

# 回滚:只把 12182 后面的 ITM 后端换回 BLIP2
ITM_BACKEND=blip2 bash scripts/launch_vlm_servers_jy.sh 0
```

回滚:

```bash
tmux kill-session -t <vlm_servers_session>
ITM_BACKEND=blip2 bash scripts/launch_vlm_servers_jy.sh 0
```

---

## 11.1 非目标

- 不改 `RealityITMPolicyV2` 的导航逻辑。
- 不改 `ValueMap` 融合公式。
- 不调 `_min_confidence` / `_decision_threshold` / `sort_waypoints(..., 0.5)` 这几个几何参数。
- SigLIP2-base 已通过 2026-06-14 cat_demo 单集验收并设为默认;so400m 不在本次默认范围内。
- 不在 `vlfm_pip` 里强行升级 `transformers` / `torch`,避免把 BLIP2/LAVIS 环境污染坏。

---

## 11.2 规划

### Phase 0 - 建保护分支与现场确认

目标:确认本次改动只落在新增服务和启动脚本,不碰导航主链路。

动作:

1. 新建分支,例如 `codex/siglip2-itm-backend`。
2. 记录 `git status --short`,不动已有未跟踪/用户现场文件,例如 `lockfiles/`。
3. 确认当前接线:
   - `itm_policy.py` 只实例化 `BLIP2ITMClient`。
   - `blip2itm.py` 的 server route 是 `/blip2itm`。
   - `launch_vlm_servers_jy.sh` 中 12182 pane 目前固定 `python -m vlfm.vlm.blip2itm`。

产物:

- 分支。
- 一条短记录:当前 BLIP2 默认启动命令和端口。

验收:

- `git diff` 没有 policy/value map/reality 配置变化。

### Phase 1 - 新增 SigLIP2 兼容服务

目标:新增 `vlfm/vlm/siglip2itm.py`,但不接入默认启动。

接口要求:

```python
class SigLIP2ITM:
    def cosine(self, image: np.ndarray, txt: str) -> float:
        ...

class SigLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        ...
```

服务要求:

- Flask route 仍注册为 `name="blip2itm"`。
- payload 仍读 `image` 和 `txt`。
- response 仍是 `{"response": float}`。
- image 输入继续接受 `server_wrapper.str_to_image` 解出来的 `np.ndarray`。
- 支持 fp16 CUDA 加载;CPU 仅作为兜底,不作为性能目标。
- 模型名通过环境变量配置,例如 `SIGLIP_MODEL_ID`,不要写死到 policy。

实现要点:

1. 使用 Hugging Face `AutoModel` / `AutoProcessor` 或 SigLIP 专用类加载模型。
2. image/text 分别编码,取 normalized embedding 后做 cosine。
3. `torch.inference_mode()` 包住推理。
4. 输出强制 `float(cosine.item())`,避免 JSON 序列化 numpy / torch 标量出错。
5. 启动时打印模型名、device、dtype,便于复盘显存与环境。

产物:

- `vlfm/vlm/siglip2itm.py`

验收:

```bash
python -m py_compile vlfm/vlm/siglip2itm.py
```

### Phase 2 - 启动脚本增加显式开关

目标:让同一套脚本可启动 BLIP2 或 SigLIP2,默认仍 BLIP2。

建议变量:

```bash
ITM_BACKEND="${ITM_BACKEND:-blip2}"
SIGLIP_CONDA_ENV="${SIGLIP_CONDA_ENV:-siglip2_itm}"
SIGLIP_MODEL_ID="${SIGLIP_MODEL_ID:-<待 smoke test 后确认>}"
```

脚本逻辑:

```bash
case "${ITM_BACKEND}" in
  blip2)
    tmux send-keys ... "${prefix} && python -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
    ;;
  siglip2)
    tmux send-keys ... "${siglip_prefix} && python -m vlfm.vlm.siglip2itm --port ${BLIP2ITM_PORT}" C-m
    ;;
  *)
    echo "Unknown ITM_BACKEND=${ITM_BACKEND}; expected blip2 or siglip2"
    exit 1
    ;;
esac
```

隔离原则:

- BLIP2 pane 继续用 `CONDA_ENV=vlfm_pip`。
- SigLIP2 pane 默认用 `SIGLIP_CONDA_ENV=siglip2_itm`。
- SigLIP2 pane 加 `PYTHONPATH=${REPO_DIR}`,这样不必把整个 VLFM pip 装进新 env。
- 其他 3 个 VLM 服务(GroundingDINO / SAM / YOLO)不动。

产物:

- `scripts/launch_vlm_servers_jy.sh` 增加 `ITM_BACKEND` 分支。

验收:

```bash
# 默认行为必须仍显示/启动 blip2itm
bash -n scripts/launch_vlm_servers_jy.sh
```

### Phase 3 - 纯空跑 smoke test

目标:先不接导航,只验证服务协议、显存、延迟、分数分布。

动作:

1. 启动 SigLIP2 后端:

   ```bash
   ITM_BACKEND=siglip2 bash scripts/launch_vlm_servers_jy.sh 0
   ```

2. 用现有 `BLIP2ITMClient` 打请求:

   ```python
   from vlfm.vlm.blip2itm import BLIP2ITMClient
   client = BLIP2ITMClient(port=12182)
   score = client.cosine(rgb, "Seems like there is a cat ahead.")
   ```

3. 记录:
   - 模型加载后的 `nvidia-smi` resident memory。
   - 首次请求耗时。
   - 稳态请求耗时。
   - 同一张图上 cat / chair / refrigerator / random prompt 的分数排序。
   - 同一文本在 3-5 张图上的分数范围。

产物:

- 一份 smoke test 记录,可以先写进临时日志或 PR 描述。

验收:

- `/blip2itm` 返回 HTTP 200。
- response 可被 `float(response["response"])` 解析。
- 连续请求 10 次无服务崩溃。
- 显存比 BLIP2-ITM 明显下降,否则没有替换收益。
- 对明显正例/负例图片,排序方向符合直觉。

### Phase 4 - 小规模导航 A/B

目标:确认 value map 排序没有明显退化。

动作:

1. BLIP2 和 SigLIP2 跑同一批 cat episode。
2. 保存 value map 可视化和最终轨迹。
3. 优先看猫方向是否被排到 frontier 前列,而不是盯绝对 cosine 数值。
4. 若 cat 通过,再扩展到 chair / refrigerator 这类 COCO 常见目标。

建议样本:

- cat:3-5 条 episode。
- chair:3 条 episode。
- refrigerator:3 条 episode。

判据:

- SigLIP2 成功 episode 至少不明显少于 BLIP2。
- value map 热区大体落在目标方向,不能稳定偏向空墙/走廊。
- 失败案例能从日志区分是 VLM 排序问题、检测问题、还是 PointNav/地图问题。

---

## 11.3 可能坑与对应解决方案

| 坑 | 现象 | 原因 | 解决方案 | 验收 |
|---|---|---|---|---|
| 污染 BLIP2 环境 | BLIP2/LAVIS 起不来,或 `transformers` 版本冲突 | 当前 `pyproject.toml` pin `transformers == 4.26.0`;SigLIP2 往往需要较新的 HF 组件 | SigLIP2 用独立 `SIGLIP_CONDA_ENV`;不要在 `vlfm_pip` 升级核心依赖 | 默认 `bash scripts/launch_vlm_servers_jy.sh 0` 仍能启动 BLIP2 |
| 路由不兼容 | policy 请求 12182 超时或 404 | 新服务 route 注册成了 `/siglip2itm` | 新服务仍 `host_model(..., name="blip2itm", port=12182)` | `BLIP2ITMClient(port=12182).cosine(...)` 可直接打通 |
| 返回格式不兼容 | client `float(response["response"])` 报错 | 返回了 list/dict/torch scalar | 强制返回 `{"response": float_value}` | 连续 10 次请求均返回 Python float |
| RGB/BGR 或 PIL 转换错误 | 分数排序很怪,正例低于负例 | `server_wrapper` 传的是 cv2 解码后的 ndarray;模型预处理通常按 PIL/RGB 预期 | 在 SigLIP2 服务里明确 `Image.fromarray(image).convert("RGB")`;必要时用保存帧肉眼确认颜色 | 同图 cat/chair/fridge prompt 排序符合肉眼直觉 |
| embedding 没归一化 | cosine 数值范围异常,不同 prompt 差异很大 | 直接点积而不是 cosine | image/text features 做 L2 normalize 后再乘 | 分数落在合理 cosine 区间,无大幅爆值 |
| 文本 prompt 与 BLIP2 习惯不同 | 排序变弱或过度弥散 | SigLIP2 是双塔全局对齐,对模板词可能敏感 | 保留原 prompt;另做只读对照:原 prompt vs `"a photo of a target_object"`;通过配置选择,不硬改 policy | cat episode 中目标方向排名稳定靠前 |
| 分数值域变了导致误判 | debug 里百分比不好看,热力图颜色变淡/过曝 | 绝对 cosine 分布不同 | 不调导航阈值;如有需要只调可视化色阶或记录分布 | frontier argmax 行为正常;可视化只是外观问题 |
| SigLIP2 响应更弥散 | value map 热区不够聚焦 | 双塔全局池化不等同 BLIP2 Q-Former ITC | 用 episode 目检 value map;必要时保留 BLIP2 默认,或只作为可选后端 | 目标方向不稳定时不切默认 |
| 首次请求慢 | 第一次 cosine 超时 | 模型 lazy init / CUDA warmup | 服务启动后做一次 warmup;client timeout 暂不改,先在 smoke 里预热 | 第二次以后稳态耗时达标 |
| HF 下载不稳定 | 首次启动卡在模型下载 | 网络/镜像问题 | 支持 `HF_ENDPOINT`;允许预下载到本地 cache;模型 id 通过 env 配置 | 离线重启可复用 cache |
| 多进程抢 GPU | VLM card 显存峰值仍高 | 四个 VLM + nav 共卡 | smoke test 记录空载/加载后/推理峰值;必要时把 SigLIP2 单独放卡测 | 替换后 resident memory 相比 BLIP2 有明确下降 |
| 回滚不彻底 | 以为回滚了但仍在跑 SigLIP2 | tmux 老 session 未杀 | 回滚步骤固定:杀旧 session 后用默认命令重启 | `tmux capture-pane` 或启动日志显示 `vlfm.vlm.blip2itm` |

---

## 11.4 详细验收清单

### 代码验收

- `git diff --stat` 只包含:
  - 新增 `vlfm/vlm/siglip2itm.py`
  - 修改 `scripts/launch_vlm_servers_jy.sh`
  - 可选:新增 smoke test 脚本/文档记录
- `vlfm/policy/itm_policy.py` 无改动。
- `vlfm/mapping/value_map.py` 无改动。
- `config/experiments/reality.yaml` 无改动。
- `bash -n scripts/launch_vlm_servers_jy.sh` 通过。
- `python -m py_compile vlfm/vlm/siglip2itm.py` 通过。

### 默认回归验收

默认命令必须仍走 BLIP2:

```bash
bash scripts/launch_vlm_servers_jy.sh 0
```

验收点:

- 12182 pane 日志显示加载 `vlfm.vlm.blip2itm`。
- `BLIP2ITMClient(port=12182).cosine(...)` 正常返回。
- 原 cat demo / reality 启动方式无需新增环境变量。

### SigLIP2 服务验收

显式命令才走 SigLIP2:

```bash
ITM_BACKEND=siglip2 bash scripts/launch_vlm_servers_jy.sh 0
```

验收点:

- 12182 pane 日志显示模型名、device、dtype。
- `/blip2itm` HTTP 200。
- 10 次 cosine 请求无崩溃。
- 显存低于 BLIP2-ITM 基线,且稳定请求延迟可接受。
- 正负样例排序符合直觉。

### 导航验收

最小通过:

- cat episode 3-5 条,SigLIP2 不出现系统性反向排序。
- value map 热区与目标方向大体一致。
- 未出现因阈值值域变化导致的"完全不探索/乱停"。

推荐通过:

- cat / chair / refrigerator 各 3 条。
- SigLIP2 成功数不明显低于 BLIP2。
- 至少保留 2-3 个失败案例截图或日志,用于判断是否值得继续优化 prompt/模型。

切默认前门槛:

- 连续两批 episode 没有明显退化。
- 显存收益足以覆盖替换风险。
- 回滚命令经实测有效。

---

## 11.5 批准后的执行顺序

建议按下面顺序提交,每一步都可停:

1. **Commit A:新增 SigLIP2 兼容服务**
   - 只加 `vlfm/vlm/siglip2itm.py`
   - 不改启动脚本
   - 可单独 py_compile

2. **Commit B:启动脚本加 `ITM_BACKEND`**
   - 默认 `blip2`
   - 显式 `siglip2`
   - `bash -n` 验收

3. **Commit C:smoke test 记录/小工具**
   - 记录显存、延迟、分数分布
   - 不接导航

4. **Commit D:导航 A/B 结果**
   - 只记录结果或小幅调整 SigLIP2 prompt/model env 默认
   - 仍不切生产默认

---

## 11.6 最小回滚手册

场景一:SigLIP2 服务效果差,但代码保留。

```bash
tmux kill-session -t <vlm_servers_session>
bash scripts/launch_vlm_servers_jy.sh 0
```

场景二:要完全回到旧代码。

```bash
git switch main
bash scripts/launch_vlm_servers_jy.sh 0
```

场景三:SigLIP2 env 装坏。

- 不修 `vlfm_pip`。
- 直接删除或重建 `siglip2_itm` env。
- 因为默认 BLIP2 不依赖该 env,生产链路不受影响。

---

## 11.7 最终决策建议

批准实现时只批准到 **可选后端 + smoke test + 小规模 A/B**,不要批准"直接替换默认"。

如果 smoke test 显示:

- 显存明显下降;
- 请求延迟可接受;
- cat/chair/fridge 排序正常;
- value map 目检不退化;

再考虑把 `ITM_BACKEND=siglip2` 写进某个实验脚本或新建 `launch_vlm_servers_siglip2.sh`。在此之前,生产默认保持 BLIP2。

---

## 11.8 2026-06-14 当前 smoke 记录

### 本机模型与环境

- SigLIP2 env:`/data/jinsong.yuan/miniconda3/envs/siglip2_itm`
- 本地模型:`/data/jinsong.yuan/siglip2-base-patch16-384`
- 已确认模型文件:
  - `config.json`
  - `preprocessor_config.json`
  - `tokenizer.json`
  - `tokenizer.model`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  - `model.safetensors`(约 1.50 GB,408 个 tensor,可被 `safetensors.safe_open` 打开)

`google/siglip2-so400m-patch14-384` 此前下载卡住,当前 smoke 改用 base 模型。`scripts/launch_vlm_servers_jy.sh` 已改为优先使用本地 base 目录;目录不存在时才退回远端 `google/siglip2-base-patch16-384`。

### 已修复问题

- Transformers 5 的 `get_image_features` / `get_text_features` 返回 `BaseModelOutputWithPooling`,不是裸 tensor;已兼容 `pooler_output`。
- OpenCV 4.13 不接受 JPEG quality float;`server_wrapper.image_to_str` 已把 quality 转成 int。
- Flask threaded worker 下 CUDA 推理从约 20 ms 退化到约 840 ms;SigLIP2 服务改为 `host_model(..., threaded=False)`。

### smoke 结果

- `python -m py_compile vlfm/vlm/siglip2itm.py vlfm/vlm/server_wrapper.py` 通过。
- `bash -n scripts/launch_vlm_servers_jy.sh` 通过。
- 直连 `SigLIP2ITM`:
  - 加载约 4.5 s。
  - PyTorch 侧显存约 764 MiB,峰值约 775 MiB。
  - 单次 cosine 稳态约 19-20 ms。
- HTTP `/blip2itm`:
  - 12192 端口 raw `requests.post` 稳态约 23-36 ms。
  - 用原 `BLIP2ITMClient(port=12192)` 连续 10 次请求均成功,端到端均值约 95 ms。
  - GPU0 resident 从约 7567 MiB 到约 8820 MiB;SigLIP2 进程 resident 约 1248 MiB。
- 分数方向弱验收:
  - 沙发/楼梯/台灯画面中 sofa/stairs/lamp 分数靠前。
  - 厨房/冰箱/椅子画面中 kitchen/refrigerator/chair 分数靠前,cat/random 靠后。

### 未完成

- 还没有接 12182 替换正在运行的 BLIP2 服务。
- 还没有跑 cat/chair/refrigerator 小批导航 A/B。
- 当前排序只用历史 RGB 帧做弱验收;切默认前仍需 value map 目检和 episode 成功率对比。

---

## 11.9 2026-06-14 持久化决定

### 结论

SigLIP2-base 已批准持久化为 `scripts/launch_vlm_servers_jy.sh` 的默认 ITM 后端:

```bash
bash scripts/launch_vlm_servers_jy.sh 0
```

默认会在 12182 的 `/blip2itm` 后面启动 `vlfm.vlm.siglip2itm`,下游 `BLIP2ITMClient` / policy / value map 不改。回滚 BLIP2:

```bash
ITM_BACKEND=blip2 bash scripts/launch_vlm_servers_jy.sh 0
```

本次一起固化的默认检测栈是 `data/yolo26l.engine`,其 engine metadata 为 `half=true,int8=false`,即 YOLO26L TensorRT FP16。

### 验收记录

- 完整 cat_demo 单集成功:
  - video:`/data/jinsong.yuan/vlfm-demo/vlfm/outputs/siglip2_cat_full_habgpu1_20260614_165653/video/episode=0-ckpt=0-distance_to_goal=0.30-success=1.00-spl=0.39-soft_spl=0.38-distance_to_goal_reward=-0.00-traveled_stairs=0.00-yaw=-120.00-target_detected=1.00-stop_called=1.00-start_yaw=2.06.mp4`
  - `success=1.0000`
  - `target_detected=1.0000`
  - `stop_called=1.0000`
  - `distance_to_goal=0.3005`
  - `spl=0.3913`
- 非冻结校验:视频 191 帧,191 个唯一帧。
- 正确 GPU 拓扑:Habitat renderer 单独放物理 GPU1;SigLIP2/VLM/torch 放物理 GPU0。不要把 Habitat EGL renderer 和 VLM 服务放同一张卡。

### SigLIP2-base 显存

本机测量命令为独立 `SigLIP2ITM` 加载 + warmup + 12 次 cosine:

- `torch.cuda.max_memory_allocated`:775 MiB
- `torch.cuda.max_memory_reserved`:800 MiB
- `nvidia-smi` 进程 resident 峰值:1248 MiB

日常容量规划按 `nvidia-smi` resident 峰值约 **1.25 GiB** 记账。

---

## 11.10 so400m 可行性审核(等待批准)

### 当前状态

- 本地尚无完整 so400m 权重。
- HF cache 里只有:
  - `config.json`
  - incomplete 权重:`576,716,800` bytes
- 远端 `google/siglip2-so400m-patch14-384/model.safetensors`:
  - `Content-Length:4544143072` bytes(约 4.23 GiB)
  - 支持 `Accept-Ranges: bytes`,可以用 `wget -c` 断点续传。

### 模型规模对比

| 模型 | text/vision 层数 | hidden | patch | 权重文件 |
|---|---:|---:|---:|---:|
| base-patch16-384 | 12 / 12 | 768 | 16 | 1.50 GB |
| so400m-patch14-384 | 27 / 27 | 1152 | 14 | 4.54 GB |

### 预计显存

按 base 实测 resident 1.25 GiB、权重文件体积约 3.0 倍估算:

- 保守预计 so400m SigLIP2 服务 resident:约 3.0-4.0 GiB。
- 在当前 GPU0 约 7.6 GiB VLM 基线下,总占用预计约 10.5-11.6 GiB,远低于 H20 143 GiB,容量可行。
- 风险主要不在容量,而在下载稳定性、延迟和排序是否更好。

### 风险与建议

- 下载风险:此前 so400m 下载卡在约 550 MiB;必须用 `wget -c -L` 直链续传,不要依赖 HF 默认下载器。
- 延迟风险:模型约 3 倍大,预计 cosine 延迟会从 base 的服务内约 20 ms 增到约 50-80 ms,端到端仍大概率可接受,但需实测。
- 效果风险:so400m 可能更强,但不保证 cat_demo / frontier 排序必然优于 base;必须跑同一条 cat_demo 和至少 3 条 cat episode 对照。

### 建议执行闸门

不直接切默认。获批后按顺序执行:

1. 用 `wget -c -L` 下载完整 so400m 到本地目录。
2. safetensors 完整性校验。
3. 12192 smoke:显存、10 次请求、排序方向。
4. Habitat 空卡拓扑下跑完整 cat_demo。
5. 只有成功率和视频表现不差于 base,才考虑把 `SIGLIP_MODEL_ID` 默认改成 so400m。

---

## 11.11 2026-06-14 SigLIP2 TensorRT 固化记录

### 产物

- ONNX:
  - `data/siglip2_vision_b16_384.onnx` 约 356 MiB,`pixel_values[1,3,384,384] -> image_embeds[1,768]`
  - `data/siglip2_text_b16.onnx` 约 1.1 GiB,`input_ids[1,64] -> text_embeds[1,768]`
- TensorRT FP16 engine:
  - `data/siglip2_vision_b16_384_fp16.engine` 约 188 MiB
  - `data/siglip2_text_b16_fp16.engine` 约 566 MiB
- COCO-80 text table:
  - `data/siglip2_text_coco80_fp16.npy` 约 124 KiB
  - `data/siglip2_text_coco80_fp16_meta.json`

### 本轮修复

- `scripts/siglip2_trt/build_engine.py` 兼容 TensorRT 10 的 `IHostMemory`:用 `memoryview(serialized)` 写文件,用文件大小打印 engine 体积。
- `scripts/siglip2_trt/export_onnx.py` 显式抽取 HF `BaseModelOutputWithPooling.pooler_output`,避免导出 token grid 等多余 output。重导后的 ONNX 均为单输出 `[1,768]`。
- `vlfm/vlm/siglip2itm.py` 增加 `_TRTRunner`:
  - TensorRT 10 name-based I/O
  - `cuda-python` H2D/D2H
  - input shape 校验、固定 shape 校验、`execute_async_v3` 失败检查
  - 兼容旧多输出 engine:优先返回唯一 rank-2 pooled embedding
- 新增 `scripts/siglip2_trt/smoke_test.py`,支持两阶段验收:
  - `siglip2_itm` 生成 torch reference
  - `siglip2_itm` 或 `yolo_trt` 加载 TRT engine 做数值对比

### 环境记录

- `siglip2_itm` 已安装 `tensorrt-cu12==10.9.0.34`:
  - 本地 cache 复用 `tensorrt_cu12` / `tensorrt_cu12_libs`
  - 仅下载 `tensorrt_cu12_bindings-10.9.0.34-cp310...whl`
- `yolo_trt` 也可加载同版本 engine；本轮最终 smoke 用 `siglip2_itm` 和 GPU1。

### 验收命令与结果

生成 torch reference:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  /data/jinsong.yuan/miniconda3/envs/siglip2_itm/bin/python \
  scripts/siglip2_trt/smoke_test.py \
  --mode torch-ref \
  --model-id /data/jinsong.yuan/siglip2-base-patch16-384 \
  --ref outputs/siglip2_trt_smoke/ref.npz
```

结果:`torch cosine=0.051808`。

TRT 数值对比:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  /data/jinsong.yuan/miniconda3/envs/siglip2_itm/bin/python \
  scripts/siglip2_trt/smoke_test.py \
  --mode trt-check \
  --ref outputs/siglip2_trt_smoke/ref.npz
```

结果:

- image embedding cosine vs torch:`0.999999`
- text embedding cosine vs torch:`1.000000`
- final cosine:`torch=0.051808`,`trt=0.051855`,`abs_diff=0.000047`
- text table path:`table_score_abs_diff=0.000009`

服务类 runtime 分支:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  SIGLIP_VISION_ENGINE=data/siglip2_vision_b16_384_fp16.engine \
  SIGLIP_TEXT_ENGINE=data/siglip2_text_b16_fp16.engine \
  SIGLIP_TEXT_CACHE=0 \
  /data/jinsong.yuan/miniconda3/envs/siglip2_itm/bin/python - <<'PY'
from scripts.siglip2_trt.smoke_test import _demo_image, DEFAULT_PROMPT
from vlfm.vlm.siglip2itm import SigLIP2ITM
itm = SigLIP2ITM(model_id="/data/jinsong.yuan/siglip2-base-patch16-384")
print(itm.cosine(_demo_image(), DEFAULT_PROMPT))
PY
```

结果:`0.051849`,与 TRT smoke 一致。

### 剩余边界

当前 `SigLIP2ITM` 服务类仍会加载 HF model / processor,即使启用了 TRT engine 或 text table；本轮已经验证了 tower engine binding 和数值等价,但“完全不加载 torch/HF 模型的 edge runtime”仍是下一步单独 refactor。
