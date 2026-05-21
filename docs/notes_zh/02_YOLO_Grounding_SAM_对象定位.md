# 02 · YOLO + GroundingDINO + SAM：从 RGB 到"最近目标点"

> 关键文件：
> - `vlfm/vlm/yolov7.py`、`vlfm/vlm/grounding_dino.py`、`vlfm/vlm/sam.py`
> - `vlfm/mapping/object_point_cloud_map.py`
> - `vlfm/policy/base_objectnav_policy.py:_update_object_map / _get_object_detections / _get_target_object_location`

## 2.1 三个模型不是平行投票，是流水线

```
                    ┌─── target 在 COCO 类里 & _load_yolo=True ───► YOLOv7Client.predict(rgb)
RGB(H,W,3) ─────────┤
                    └─── 否则 ───► GroundingDINOClient.predict(rgb, caption=非COCO提示)

  ↓  ObjectDetections {boxes(N,4), logits(N,), phrases(N,)}
  ↓  filter_by_class(target) → filter_by_conf(0.8 / 0.4)
  ↓  (兜底：COCO 走 YOLO 0 检出时再用 Grounding 跑一次)

  per detection:
        bbox(4,) ──► MobileSAMClient.segment_bbox(rgb, bbox)
                          └─► mask(H,W) bool
        (可选) BLIP2-VQA "Is this a chair?" → 不是 yes 就丢
        mask + depth + intrinsics ──► get_point_cloud → (M,3) 点云
                                       └─► DBSCAN 保留最大簇 → (M',3)
                                       └─► within_range 标注（0/1 或 随机数）
                                       └─► transform_points 到 episodic frame
                                       └─► 累加到 ObjectPointCloudMap.clouds[target]
```

## 2.2 检测层：YOLO vs GroundingDINO

`vlfm/policy/base_objectnav_policy.py:221-241`

```221:241:vlfm/policy/base_objectnav_policy.py
def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
    target_classes = self._target_object.split("|")
    has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
    has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

    detections = (
        self._coco_object_detector.predict(img)
        if has_coco
        else self._object_detector.predict(img, caption=self._non_coco_caption)
    )
    detections.filter_by_class(target_classes)
    det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
    detections.filter_by_conf(det_conf_threshold)

    if has_coco and has_non_coco and detections.num_detections == 0:
        detections = self._object_detector.predict(img, caption=self._non_coco_caption)
        detections.filter_by_class(target_classes)
        detections.filter_by_conf(self._non_coco_threshold)

    return detections
```

要点：
- `target_object` 形如 `"chair"` 或 `"table|dining table|coffee table|side table|desk"`（用 `|` 表多个别名同义类，见 `habitat_policies.py:MP3D_ID_TO_NAME`）。
- 阈值：YOLO 默认 `coco_threshold=0.8`，Grounding 默认 `non_coco_threshold=0.4`。
- Reality 模式 `_load_yolo=False`，**永远只走 Grounding**（你在 Spot 上看到的就是 Grounding）。

### 2.2.1 YOLOv7

`vlfm/vlm/yolov7.py:50-110`

- 输入：RGB `np.ndarray (H,W,3) uint8`。
- 内部 letterbox 到 `(640, 0.7·640) = (640, 448)`，半精度 fp16 推理（cuda）。
- 输出 `ObjectDetections`：
  - `boxes`：`Tensor (N,4)`，xyxy **归一化** 到 [0,1]。
  - `logits`：`Tensor (N,)`，置信度。
  - `phrases`：长度 N 的字符串列表，来自 `COCO_CLASSES`。
- Flask RPC：端口 `YOLOV7_PORT=12184`，路径 `/yolov7`。

### 2.2.2 GroundingDINO

`vlfm/vlm/grounding_dino.py:38-74`

- 输入 + `caption` 字符串（句号分隔的类别列表）。
- HM3D 时 `_non_coco_caption` 是空字符串（因为 HM3D 的 6 类全在 COCO，所以 hard-code 走 YOLO）。
- MP3D 时 `habitat_policies.py:136` 会拼出 `"chair . table . dining table . ... ."`。
- 输出和 YOLO 一致的 `ObjectDetections`，但 `boxes` 默认是 `cxcywh` 归一化，构造器内部 `box_convert` 成 `xyxy`。
- 内部 `filter_by_class` 严格保留 phrase ∈ caption 拆出来的类。
- Flask RPC：`GROUNDING_DINO_PORT=12181`，路径 `/gdino`。

### 2.2.3 ObjectDetections 数据格式

`vlfm/vlm/detections.py`

```python
{
  "boxes":   torch.Tensor (N, 4),  # xyxy normalized 0~1
  "logits":  torch.Tensor (N,),    # 0~1
  "phrases": list[str] of len N    # 类名
}
```

## 2.3 SAM：bbox → 像素级 mask

`vlfm/policy/base_objectnav_policy.py:319-321`

```python
for idx in range(len(detections.logits)):
    bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
    object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())
```

- `MobileSAMClient.segment_bbox`：把归一化框反归一化成像素坐标，扔给 MobileSAM Predictor，**返回与 RGB 同分辨率 `(H,W)` 的 bool mask**。
- Flask RPC：`SAM_PORT=12183`，路径 `/mobile_sam`，传输用 `bool_arr_to_str` base64 编码。

## 2.4 可选：BLIP2 VQA 二次确认

`vlfm/policy/base_objectnav_policy.py:326-335`

```python
if self._use_vqa:
    contours, _ = cv2.findContours(object_mask, ...)
    annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
    question = f"Question: {self._vqa_prompt}"
    if not detections.phrases[idx].endswith("ing"):
        question += "a "
    question += detections.phrases[idx] + "? Answer:"
    answer = self._vqa.ask(annotated_rgb, question)
    if not answer.lower().startswith("yes"):
        continue
```

- 默认 `use_vqa=False`（论文主表关闭，ablation 才开）。
- 用 BLIP2 (`vlfm/vlm/blip2.py` 的完整 captioning/VQA 模型，端口 12185)。

## 2.5 mask → 点云 → ObjectPointCloudMap

`vlfm/mapping/object_point_cloud_map.py:_extract_object_cloud / update_map`

### 2.5.1 抠点云

```143:163:vlfm/mapping/object_point_cloud_map.py
def _extract_object_cloud(self, depth, object_mask, min_depth, max_depth, fx, fy):
    final_mask = object_mask * 255
    final_mask = cv2.erode(final_mask, None, iterations=self._erosion_size)
    valid_depth = depth.copy()
    valid_depth[valid_depth == 0] = 1  # 把空洞当远处
    valid_depth = valid_depth * (max_depth - min_depth) + min_depth
    cloud = get_point_cloud(valid_depth, final_mask, fx, fy)
    cloud = get_random_subarray(cloud, 5000)
    if self.use_dbscan:
        cloud = open3d_dbscan_filtering(cloud)
    return cloud
```

- `_erosion_size = 5`：mask 先腐蚀 5 次（减边缘噪声）。
- `get_point_cloud`（`utils/geometry_utils.py:216`）：
  - `v, u = np.where(mask)` 取 mask 内像素索引
  - `z = depth[v,u]`，`x=(u-W/2)·z/fx`，`y=(v-H/2)·z/fy`
  - 输出按 `(z, -x, -y)` 排（**相机坐标系：前 / 右 / 下**），shape `(M, 3)`。
- 随机降采样到 ≤ 5000 点。
- `open3d_dbscan_filtering` (eps=0.2, min_points=100)：保留**最大簇**，过滤离群。

### 2.5.2 标注 within_range（"5% margin"）

```51:63:vlfm/mapping/object_point_cloud_map.py
if too_offset(object_mask):
    within_range = np.ones_like(local_cloud[:, 0]) * np.random.rand()
else:
    within_range = (local_cloud[:, 0] <= max_depth * 0.95) * 1.0
    within_range = within_range.astype(np.float32)
    within_range[within_range == 0] = np.random.rand()
global_cloud = transform_points(tf_camera_to_episodic, local_cloud)
global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1)
```

- 每个点附加第 4 列 `within_range`：
  - `1.0` → 当时确认在 5 m·0.95=4.75 m 内；
  - 否则给一个**随机数**（这一帧所有不可信点共享同一随机数，做 "detection id"）。
- `too_offset`(`object_point_cloud_map.py:269`)：如果 bbox 整个在图像左 1/3 或右 1/3，且贴边 5% 内 → 整组点都标随机数（视为不可靠）。
- 点云在 `transform_points` 把它从相机系送到 episodic 系。

### 2.5.3 距离门控（避免太近）

```65:70:vlfm/mapping/object_point_cloud_map.py
curr_position = tf_camera_to_episodic[:3, 3]
closest_point = self._get_closest_point(global_cloud, curr_position)
dist = np.linalg.norm(closest_point[:3] - curr_position)
if dist < 1.0:
    return
```

- < 1 m 不写入（太近不可信，深度抖动大）。

### 2.5.4 "假阳性消除" `update_explored`

```102:132:vlfm/mapping/object_point_cloud_map.py
def update_explored(self, tf_camera_to_episodic, max_depth, cone_fov):
    ...
    for obj in self.clouds:
        within_range = within_fov_cone(camera_coordinates, camera_yaw, cone_fov, max_depth*0.5, self.clouds[obj])
        range_ids = set(within_range[..., -1].tolist())
        for range_id in range_ids:
            if range_id == 1:
                continue
            self.clouds[obj] = self.clouds[obj][self.clouds[obj][..., -1] != range_id]
```

- 如果某 detection-id（随机数）原本被标 0（不可信），现在却落在相机视锥 `max_depth/2` 内但**仍没被升格为 1**，说明走近了也没确认到 → 删掉这个 detection 的所有点。
- 这是 VLFM 论文里说的"消除短暂误报"机制。

## 2.6 选目标点：`get_best_object`

```77:100:vlfm/mapping/object_point_cloud_map.py
def get_best_object(self, target_class: str, curr_position: np.ndarray) -> np.ndarray:
    target_cloud = self.get_target_cloud(target_class)
    closest_point_2d = self._get_closest_point(target_cloud, curr_position)[:2]
    if self.last_target_coord is None:
        self.last_target_coord = closest_point_2d
    else:
        delta_dist = np.linalg.norm(closest_point_2d - self.last_target_coord)
        if delta_dist < 0.1:
            return self.last_target_coord
        elif delta_dist < 0.5 and np.linalg.norm(curr_position - closest_point_2d) > 2.0:
            return self.last_target_coord
        else:
            self.last_target_coord = closest_point_2d
    return self.last_target_coord
```

- `get_target_cloud`：如果存在 `within_range==1` 的点，优先只看这些；否则用全部。
- `_get_closest_point` (DBSCAN 模式)：取离 `curr_position` 欧氏距离最小的点。
- **抖动抑制**：
  - 新最近点与上次差 < 0.1 m → 沿用上次（避免 jitter）
  - 差 < 0.5 m 且距离机器人 > 2 m → 也沿用上次（远处微调不必跟）
  - 否则更新

> 这就是你说的"走向最近点"。在 `_pointnav` 里再把这个 (x,y) 转成 (ρ,θ)，喂给 PointNav RNN policy。

## 2.7 阈值/超参速查

| 名称 | 值 | 来源 |
| --- | --- | --- |
| `coco_threshold`(YOLO) | 0.8 | `VLFMConfig` |
| `non_coco_threshold`(Grounding) | 0.4 | `VLFMConfig` |
| GroundingDINO `box_threshold` | 0.35 | `grounding_dino.py:30` |
| GroundingDINO `text_threshold` | 0.25 | `grounding_dino.py:30` |
| SAM `multimask_output` | False | `sam.py:55` |
| 点云 max 点数 | 5000 | `_extract_object_cloud` |
| DBSCAN `eps` / `min_points` | 0.2 / 100 | `open3d_dbscan_filtering` |
| within_range margin | 5%（即 `<= max_depth · 0.95`） | `object_point_cloud_map.py:56` |
| 太近不写入阈值 | 1.0 m | `object_point_cloud_map.py:68` |
| 假阳性消除范围 | `max_depth · 0.5` | `object_point_cloud_map.py:123` |
| 抖动抑制 | 0.1 m / 0.5 m & 2 m | `get_best_object` |
| 抓到目标后停止半径 | 0.9 m | `pointnav_stop_radius` |

## 2.8 直觉修正

| 你说 | 实际 |
| --- | --- |
| "YOLO 负责识别，Grounding 负责切割" | 都是**识别**（分类+bbox）。切割（segmentation）是 **MobileSAM** 干的。YOLO 和 Grounding 二选一兜底关系，不是同一帧同时跑。 |
| "如果 YOLO 有结果就覆盖 BLIP" | 不是覆盖 BLIP，而是 `act()` 切到 `navigate` 分支：goal 由 `ObjectPointCloudMap` 给，**完全不再看 ValueMap**。下一步 ValueMap 还是会被更新（BLIP2-ITM 照样跑），但目标点不依赖它。 |
| "走向最近点" | ✅ 对（带抖动抑制 + within_range 优先 + DBSCAN 主簇）。 |
