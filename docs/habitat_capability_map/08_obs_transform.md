# ObsTransform

ObsTransform 是 Habitat-Baselines 中在 observation 进入 policy 前做预处理的边界。它不新增环境信息，只改变已有 observation 的 shape、dtype、尺度或组织方式。

一句话：ObsTransform 是“输入矩阵整理层”。

## 常见 ObsTransform 选项

| Transform | 输入形态 | 输出形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- | --- |
| Resize | `(B, H, W, C)` 图像 | `(B, H2, W2, C)` | 把 RGB 从 480x640 变成 224x224 | policy 输入尺寸固定 |
| Center crop | `(B, H, W, C)` | `(B, cropH, cropW, C)` | 去掉图像边缘，只保留中心区域 | 减少输入尺寸，但可能丢信息 |
| Normalize RGB | uint8 或 float 图像 | normalized float tensor | 减均值、除方差 | 匹配视觉 encoder 的输入分布 |
| Depth normalization | depth float 矩阵 | 缩放后的 depth | 把米制深度变成 0 到 1 | 模型训练更稳定 |
| Channel reorder | `(B, H, W, C)` | `(B, C, H, W)` | 适配 PyTorch CNN | 改变内存和模型期望布局 |
| dtype cast | uint8 / int / float | 指定 dtype | RGB 转 float，semantic 保持 int | 保证模型和 loss 能处理 |
| Semantic nearest resize | `(B, H, W)` int | `(B, H2, W2)` int | resize semantic map | 保持类别 id 不被插值混合 |
| Frame stack | 多帧 observation | 通道或时间维拼接 | policy 看到最近 4 帧 | 提供短期动态信息 |
| Key filtering | observation dict | 只保留部分 keys | policy 只用 depth，不用 rgb | 降低输入复杂度 |
| Key rename / remap | observation dict | 新 key 名称 | 兼容某个 policy 期望 key | 解决接口命名不一致 |
| Padding / batching | 变长数据 | 固定 shape + mask | 指令 token 或候选列表补齐 | 适合 batch 训练 |

不是每个 transform 都是所有 Habitat 版本默认内置；核心边界是 Habitat-Baselines 支持把 ObservationTransformer 注册进 policy 输入管线。

## 不同 observation 的数据形态处理

| Observation | 原始形态 | 常见处理 | 效果 |
| --- | --- | --- | --- |
| RGB | `(H, W, 3)` uint8 | resize、normalize、channel reorder | 适配 CNN / transformer 输入 |
| Depth | `(H, W, 1)` float | resize、clip、normalize | 适配 PointNav 或视觉 policy |
| Semantic | `(H, W)` int | nearest resize、保持 int | 类别 id 不被破坏 |
| PointGoal | `(2,)` 或 `(3,)` float | normalize 或直接输入 | 保持目标向量 |
| ObjectGoal | scalar int | embedding 或 one-hot 由 policy 内处理 | 类别目标进入模型 |
| Instruction | string / token ids | tokenize、truncate、pad | 适配语言 encoder |
| ImageGoal | `(H, W, 3)` | resize、normalize | 与当前 RGB 形态对齐 |
| Recurrent masks | `(B, 1)` | 保持 bool / float | 控制 hidden state reset |

## Resize 细节

| 选项 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| RGB resize | `(B, H, W, 3)` 到 `(B, 224, 224, 3)` | 固定视觉 policy 输入 | 降低计算量 |
| Depth resize | `(B, H, W, 1)` 到 `(B, 224, 224, 1)` | PointNav policy 需要固定 depth 尺寸 | 保持输入契约 |
| Semantic resize | `(B, H, W)` 到 `(B, H2, W2)` | 语义图跟 RGB 对齐 | 必须用 nearest，避免 id 被混合 |
| 多 key resize | dict 中多个图像 key 同时变换 | RGB、Depth 同尺寸进入 policy | 多模态 shape 对齐 |

注意：如果 resize 后的数据还要用于几何投影，相机内参和像素尺度也要一起考虑。ObsTransform 只负责矩阵变化，不自动修正所有下游几何假设。

## Crop / Padding / Batch 例子

| 需求 | 输入形态 | 输出形态 | 效果 |
| --- | --- | --- | --- |
| 只看图像中心 | `(B, H, W, C)` | `(B, cropH, cropW, C)` | 去掉边缘，降低计算 |
| 保持方形输入 | 非方形图像 | 方形 crop 或 pad | 适配固定模型 |
| 指令长度不同 | token 序列长度不同 | `(B, L)` + mask | 能组成 batch |
| 多个 goal 数量不同 | list 长度不同 | `(B, N, D)` + mask | 让变长结构进入模型 |
| 多帧堆叠 | T 个 `(H, W, C)` | `(H, W, T*C)` 或 `(T, H, W, C)` | policy 能看到短期历史 |

## 非代码用例库

| 需求 | ObsTransform 选择 | 数据形态变化 | 效果 |
| --- | --- | --- | --- |
| 训练用统一图像尺寸 | Resize | `(H, W, C)` 到 `(224, 224, C)` | 所有场景输入一致 |
| 复用预训练视觉网络 | Normalize + channel reorder | uint8 HWC 到 float CHW | 符合模型输入约定 |
| 保持 semantic 正确 | nearest resize | int map 到 int map | 类别 id 不混乱 |
| 减少显存 | resize 到更小 H/W | 图像矩阵更小 | 训练更快，细节减少 |
| 让语言指令 batch 化 | tokenize + padding | string 到 `(L,)` token ids | 可以并行训练 |
| 比较 RGB-only 和 Depth-only | key filtering | observation dict 去掉某些 key | 做输入消融 |
| 支持 recurrent policy | masks 保持并传入 | `(B, 1)` reset mask | episode 结束时清 hidden |

## ObsTransform 与 Sensor 的边界

| 情况 | 属于 Sensor | 属于 ObsTransform |
| --- | --- | --- |
| 新增 RGB 相机 | 是 | 否 |
| 把 RGB resize | 否 | 是 |
| 新增 Semantic sensor | 是 | 否 |
| Semantic 用 nearest resize | 否 | 是 |
| 新增 Instruction | 是 | 否 |
| Instruction token padding | 否 | 是 |
| 新增局部地图 observation | 是 | 否 |
| 把局部地图 crop 到固定大小 | 否 | 是 |

## 边界

ObsTransform 可以：

- 改 shape。
- 改 dtype。
- 改数值尺度。
- 对齐多模态输入尺寸。
- 过滤或重命名 key。
- 支持 batching。

ObsTransform 不应该：

- 新增环境真值信息。
- 改任务成功条件。
- 决定下一步 action。
- 改 simulator 渲染本身。

## 什么时候改 ObsTransform

优先改 ObsTransform，当你要改变：

- 图像输入尺寸。
- 图像通道顺序。
- RGB / depth / semantic 的数值预处理。
- 语言或变长输入的 padding。
- policy 看到的 observation key 组织方式。
- 训练和评估时的 batch 输入契约。

