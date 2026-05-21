# 01 · BLIP2 + ValueMap：从余弦相似度到融合公式

> 关键文件：
> - `vlfm/vlm/blip2itm.py`（模型客户端）
> - `vlfm/mapping/value_map.py`（核心地图）
> - `vlfm/policy/itm_policy.py:_update_value_map`（调度入口）

## 1.1 整体流水线（一帧）

```
RGB (H,W,3) ──► BLIP2ITMClient.cosine(image, text)  ── 余弦标量 ∈ ~[0, 0.6]
                                  │
                                  ▼
                            np.array(cosines)  形状 (C,)   ← C = prompt 个数
                                  │
                                  ▼
Depth (H,W) + tf(4,4) + fov + min/max ──► ValueMap.update_map(values, depth, tf, ...)
                                  │
                          ┌───────┴──────────┐
                          ▼                  ▼
              _localize_new_data       _fuse_new_data
              (得到 curr_map H×H)      (写入 self._map, self._value_map)
```

## 1.2 BLIP2-ITM 余弦的具体取法

`vlfm/vlm/blip2itm.py:37-54`

```37:54:vlfm/vlm/blip2itm.py
def cosine(self, image: np.ndarray, txt: str) -> float:
    pil_img = Image.fromarray(image)
    img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
    txt = self.text_processors["eval"](txt)
    with torch.inference_mode():
        cosine = self.model({"image": img, "text_input": txt}, match_head="itc").item()
    return cosine
```

- 走的是 LAVIS `blip2_image_text_matching` 模型的 **ITC head**（Image-Text Contrastive），返回**一个浮点标量**。
- 通过 Flask 在 `BLIP2_ITM_PORT=12182` 上以 base64-PNG + 文本的方式 RPC 调用。
- 这里 **没有任何空间信息**——一张图只产生一个标量。每个 prompt 跑一次。

调度方在 `vlfm/policy/itm_policy.py:191-211`：

```191:206:vlfm/policy/itm_policy.py
def _update_value_map(self) -> None:
    all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
    cosines = [
        [
            self._itm.cosine(
                rgb,
                p.replace("target_object", self._target_object.replace("|", "/")),
            )
            for p in self._text_prompt.split(PROMPT_SEPARATOR)
        ]
        for rgb in all_rgb
    ]
    for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
        cosines, self._observations_cache["value_map_rgbd"]
    ):
        self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)
```

- `text_prompt` 默认是 `"Seems like there is a target_object ahead."`（`VLFMConfig.text_prompt`）。
- 用 `|` 分多个 prompt → 多通道 value map。**V2 默认只有 1 个 prompt（即 1 通道）**。
- `target_object` 字面替换为目标类别。`"chair|table|..."` 这种带 `|` 的目标会先 `replace("|", "/")` 给 BLIP2 看。

## 1.3 单帧的 "局部锥形 + 置信度" 怎么算

入口 `ValueMap._process_local_data`（`vlfm/mapping/value_map.py:221-286`）。

### 1.3.1 先生成"满血锥形" `_get_blank_cone_mask`

`value_map.py:321-335`

- 用 `cv2.ellipse` 画一个 `(2·size+1)²` 的圆形扇区，扇区角度 = `±fov/2`，`size = max_depth · pixels_per_meter`。
- HM3D 默认：`max_depth = 5.0m, pixels_per_meter = 20, fov = 79°`
  - `size = 100 px` → `cone_mask.shape = (201, 201)`，里面是 0/1。

### 1.3.2 升级成 "靠近光轴更可信" `_get_confidence_mask`

`value_map.py:337-355`

- 对每个像素：
  - 行偏移 = `|row - cy|`，列偏移 = `|col - cx|`
  - `angle = arctan2(vertical, horizontal)`（注意：这里 `vertical` 是列偏移，`horizontal` 是行偏移；扇形顶点朝下/上看作"光轴 = 0°"）
  - `angle = remap(angle, 0, fov/2, 0, π/2)` —— 把 `[0, fov/2]` 拉到 `[0, π/2]`
  - `confidence = cos²(angle)`，再 `remap(0,1, 0.25, 1)` —— 最低 0.25，最高 1.0
- 然后 **乘上** `cone_mask`（扇区外为 0）。结果 `adjusted_mask` 是 `(201,201) float32`，扇形里数值 ∈ [0.25, 1.0]，扇形外 0。
- 第一次算后 cache 在 `_confidence_masks[(fov, max_depth)]`。

> **这就是你说的"通道 2 的 confidence"。** 但实际它**不是 value_map 的一个通道**，而是另一张同尺寸的二维图 `self._map`，下面 1.4 会再说。

### 1.3.3 用深度图把被遮挡部分挖掉

`value_map.py:230-260`

```234:260:vlfm/mapping/value_map.py
depth_row = np.max(depth, axis=0) * (max_depth - min_depth) + min_depth

angles = np.linspace(-fov / 2, fov / 2, len(depth_row))

x = depth_row
y = depth_row * np.tan(angles)

cone_mask = self._get_confidence_mask(fov, max_depth)

x = (x * self.pixels_per_meter + cone_mask.shape[0] / 2).astype(int)
y = (y * self.pixels_per_meter + cone_mask.shape[1] / 2).astype(int)

last_row = cone_mask.shape[0] - 1
last_col = cone_mask.shape[1] - 1
start = np.array([[0, last_col]])
end = np.array([[last_row, last_col]])
contour = np.concatenate((start, np.stack((y, x), axis=1), end), axis=0)

visible_mask = cv2.drawContours(cone_mask, [contour], -1, 0, -1)
```

直觉：
- `depth` 是 `(H_img, W_img)` 归一化到 [0,1]。沿列取 `max` 得到 `(W_img,)` 的"最远可见距离"折线。
- 把每一列的 `(深度, tan(angle)*深度)` 作为锥形里"墙的位置"，串成一条折线。
- 折线之外（更远的、被墙挡住的）用 `cv2.drawContours(..., 0, -1)` 涂成 0。
- 剩下的部分保持原来的 `confidence ∈ [0.25,1]`。

📐 **本步输出**：`visible_mask`，形状 `(201, 201)` 的 float32，**这就是单帧局部 confidence 图**。

### 1.3.4 旋转 + 贴到全局大图 `_localize_new_data`

```288:319:vlfm/mapping/value_map.py
def _localize_new_data(self, depth, tf_camera_to_episodic, min_depth, max_depth, fov):
    curr_data = self._process_local_data(depth, fov, min_depth, max_depth)
    yaw = extract_yaw(tf_camera_to_episodic)
    curr_data = rotate_image(curr_data, -yaw)
    cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
    px = int(cam_x * self.pixels_per_meter) + self._episode_pixel_origin[0]
    py = int(-cam_y * self.pixels_per_meter) + self._episode_pixel_origin[1]
    curr_map = np.zeros_like(self._map)
    curr_map = place_img_in_img(curr_map, curr_data, px, py)
    return curr_map
```

- 全局大图固定 `size=1000, pixels_per_meter=20` → **50m × 50m 的 episodic 坐标系俯视图**，原点在大图中心 `(500, 500)`。
- 把刚才 `(201,201)` 的局部锥按 `yaw` 旋转、贴在大图相机当前 px 位置 → 输出 `curr_map`，形状 `(1000, 1000)`，**就是新一帧的"全局 confidence 图"** $c_{curr}$。

## 1.4 双图融合（你最关心的公式）

入口 `ValueMap._fuse_new_data`（`vlfm/mapping/value_map.py:357-429`）。两个状态：

| 状态 | shape | 含义 |
| --- | --- | --- |
| `self._map` | `(1000, 1000)` float32 | **历史 confidence** $c_{prev}$ |
| `self._value_map` | `(1000, 1000, C)` float32 | **历史 value**（每通道一个 prompt 的余弦） $v_{prev}$ |

`values` 参数是 `(C,)`，**整张新可见区域所有像素共享**这一组余弦（因为 BLIP2 只给标量）。`new_map`(=curr_map) 是 `(1000,1000)`，**新 confidence** $c_{curr}$。

### 1.4.1 同步用 ObstacleMap 强制清零未探索区（可选）

```369:375:vlfm/mapping/value_map.py
if self._obstacle_map is not None:
    explored_area = self._obstacle_map.explored_area
    new_map[explored_area == 0] = 0
    self._map[explored_area == 0] = 0
    self._value_map[explored_area == 0] *= 0
```

> 只有 `sync_explored_areas=True` 才打开（Reality 里默认开，Habitat ITMPolicyV2 默认关）。

### 1.4.2 软门控：低置信度新数据被"silence"

```397:399:vlfm/mapping/value_map.py
new_map_mask = np.logical_and(new_map < self._decision_threshold, new_map < self._map)
new_map[new_map_mask] = 0
```

- `_decision_threshold = 0.35`，`_min_confidence = 0.25`。
- 直觉：如果新观测既"信度不到 0.35"又"比历史还差"，干脆不让它写入。

### 1.4.3 模式 A：max-confidence（`use_max_confidence=True`）

```401:408:vlfm/mapping/value_map.py
if self._use_max_confidence:
    higher_new_map_mask = new_map > self._map
    self._value_map[higher_new_map_mask] = values
    self._map[higher_new_map_mask] = new_map[higher_new_map_mask]
```

- 哪些像素新置信度大于旧的 → **直接用新的 v 和 c 覆盖**。
- 这就是你说的"**没重合 → 直接写入**"的真实写法：没历史的像素 `self._map[i,j]=0`，任何 c_curr>0 都大于 0，自然会被覆盖；有历史但新观测更近/更接近光轴时也会覆盖。
- 注意：`ValueMap.__init__` 和 `ITMPolicy.__init__` 的默认参数都是 `True`，但**真实跑的 `VLFMConfig` / `vlfm_objectnav_hm3d.yaml` / `reality.yaml` 都把它显式改成 `False`**，所以下面 1.4.4 才是论文实际使用的分支。

### 1.4.4 模式 B：加权平均（论文里的标准公式，**HM3D/MP3D/Spot 实际默认**）

`VLFMConfig` 把 `use_max_confidence: bool = False`（`base_objectnav_policy.py:381`），所有正式实验都走这里：

```409:429:vlfm/mapping/value_map.py
else:
    confidence_denominator = self._map + new_map
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        weight_1 = self._map / confidence_denominator
        weight_2 = new_map / confidence_denominator

    weight_1_channeled = np.repeat(np.expand_dims(weight_1, axis=2), self._value_channels, axis=2)
    weight_2_channeled = np.repeat(np.expand_dims(weight_2, axis=2), self._value_channels, axis=2)

    self._value_map = self._value_map * weight_1_channeled + values * weight_2_channeled
    self._map = self._map * weight_1 + new_map * weight_2

    self._value_map = np.nan_to_num(self._value_map)
    self._map = np.nan_to_num(self._map)
```

写成数学：每个像素 $(r,c)$ 设 $a = c_{prev}, b = c_{curr}$，则

$$
\boxed{
\;w_a = \frac{a}{a+b},\quad w_b = \frac{b}{a+b}
\;}
$$

$$
\boxed{
\;v_{new}[k] = w_a \cdot v_{prev}[k] + w_b \cdot v_{curr}[k] \quad (\forall k=0\dots C{-}1)
\;}
$$

$$
\boxed{
\;c_{new} = w_a \cdot a + w_b \cdot b = \frac{a^2 + b^2}{a+b}
\;}
$$

> 这正是论文里的 "self-weighted average"，等价于按 $c$ 平方加权。如果 $a=b$，$c_{new}=a=b$（没增长）；如果一边远大于另一边，结果趋近大的那个。

`nan_to_num` 处理 $a=b=0$ 的像素（除 0 得 NaN）。

### 1.4.5 还有两个 ablation

- `"replace"`：新数据无条件覆盖（论文里测的 "no temporal fusion"）。
- `"equal_weighting"`：把任何 $c>0$ 的像素强行设成 1，再走加权平均，等同算术平均。

由环境变量 `MAP_FUSION_TYPE` 切换。

## 1.5 用 ValueMap 选 frontier

`ValueMap.sort_waypoints`（`value_map.py:146-187`）：

1. 把 frontier 的 xy（米）转成像素坐标。
2. 在以 `radius=0.5m → 10px` 为半径的圆内取该通道的 **median**（`pixel_value_within_radius`），返回 `(C,) tuple`。
3. 多通道时用 `_reduce_values`（V3 才有）合成单一标量：
   - 如果**所有 frontier** 的目标通道 max < `exploration_thresh` → 用第二通道（"exploration prompt"）排序；
   - 否则用第一通道（"target prompt"）排序。
4. 按降序返回 sorted frontier + sorted value。

> 这是 V3 的"target-prompt / exploration-prompt 双 prompt"机制。HM3D 默认 yaml 里 `exploration_thresh=0`，等价 V2。

## 1.6 一帧的数据形状全景（HM3D）

| 变量 | shape | dtype | 取值 |
| --- | --- | --- | --- |
| `rgb` | (480, 640, 3) | uint8 | 0–255 |
| `depth` | (480, 640) | float32 | 0–1（已归一化） |
| `cosines` | (1,) 或 (2,) | float32 | ~[0, 0.6] |
| `cone_mask` / 局部 confidence | (201, 201) | float32 | 扇内 [0.25, 1]，扇外 0 |
| `visible_mask`（挖掉障碍后） | (201, 201) | float32 | 同上但被深度截断 |
| `curr_map`（贴到 episodic 全图） | (1000, 1000) | float32 | 同上语义 |
| `self._map` | (1000, 1000) | float32 | 累积 $c_{prev}$ |
| `self._value_map` | (1000, 1000, C) | float32 | 累积 $v_{prev}$ |

> 全图 50m×50m 是因为 `size=1000, pixels_per_meter=20`。Player 模式下会扩到 `size=2000`（100m）。

## 1.7 你直觉的具体修正

| 你说 | 实际 |
| --- | --- |
| "通道 1 是 value，通道 2 是 confidence" | 二者各自是一张图：`_value_map[H,W,C]`(value) 和 `_map[H,W]`(confidence)。当 `C>1` 时，C 个通道**全部都是 value**，只是对应不同 prompt 的余弦。 |
| "锥形里靠近光轴决定 confidence" | ✅ 对，公式是 $c = \text{remap}(\cos^2\theta, 0,1, 0.25, 1)$。 |
| "去掉障碍后投到 2D" | ✅ 对，`_process_local_data` 用 depth 沿列取 max 画轮廓涂 0。 |
| "没重合 → 直接每个 3D 像素双通道" | 实际是 2D 像素（俯视图）。"没历史"等价 `self._map[i,j]=0`，融合分支自动覆盖（max-conf 模式）或权重退化为只有新数据（加权平均模式的极限）。 |
| "v_new = f(v_prev, v_curr, c_prev, c_curr)" | ✅ 对。具体公式见 1.4.4 框框。 |
