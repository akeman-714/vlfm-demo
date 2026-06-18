# 换猫 / 换 GLB → 修贴图 → 建 episode → 跑 InstanceNav(二次确认) 端到端教学

> 目标：下次换一只猫 / 换一个 glb 场景时，**照抄本文命令即可**，从 Blender 资产一路跑到远端出视频。
> 本文记录的是 2026-06-18「加奶牛猫(第二只猫) + 找奶牛猫云端二次确认」这次实跑的真实流程与踩过的坑。

约定：
- 本地 = Windows，工程在 `D:\fuxian\test\00800-TEEsavR23oF\`（放各种 glb）。
- 远端 = `train-server`（免密 ssh），仓库 `/data/jinsong.yuan/vlfm-demo/vlfm`，场景目录
  `…/data/scene_datasets/hm3d/val/00800-TEEsavR23oF/`。
- 远端跑仿真用 conda env `vlfm_pip`；ITM 用独立 env `siglip2_itm`。

---

## 0. 一图流程

```
Blender 摆位        本地清洗+改名         上传/校验/备份        远端合并(修贴图)        覆盖场景
cat.glb + house  ─► clean_cat.glb     ─► scp 到 _incoming  ─► merge_cat_into   ─► mv 覆盖
  → 导出 layout      layout.glb(改名)     md5 比对 + 备份        _scene.py            basis.glb
(贴图已损坏)                                                  (校验 SHA/AABB)
                                                                   │
   建 episode ◄───────────────────────────────────────────────────┘
build_cat_episode.py --prefix <P>  (固定起点 or 自动朝向猫)
   │
   └─► 启动 VLM 服务 + 二次确认验证器(带 key/base_url) ─► 跑 eval_cat_demo.sh ─► 出 mp4
```

---

## 1. 根因：为什么 Blender 导一圈贴图就没了

HM3D 的 `*.basis.glb` 用 **Basis Universal 压缩纹理**（`image/x-basis`）。Blender 解不了，导入即静默丢光所有房屋贴图，再导出就是个无贴图的房子。

**所以 Blender 只用来"摆位"，绝不用来导出最终场景。** 真正的合并在 glTF 二进制层做：
原始场景字节**逐字节照抄**（贴图原封不动），只把猫追加进去 —— 这就是 `merge_cat_into_scene.py` 干的事，也是"本地脚本补全 glb 贴图"的本质。

---

## 2. 资产准备（Blender，本地）

1. 新建/打开一个工程，导入 **原始 house glb**（无所谓贴图丢失）+ **猫模型 glb**（自带贴图）。
2. **只移动/旋转/缩放猫这个 object 整体**摆到你想要的位置；
   - 不要 apply/bake transform，不要编辑 mesh，不要动房子 —— 否则后面变换提取/对齐校验会(可检测地)失败。
3. 导出整个场景为一个 glb（即 `--layout`，本文叫 `twocat.glb`/`nocoverglb.glb`）。**它贴图是坏的，没关系**，它只提供"猫摆在哪"的位姿。
4. 另外单独保留那只**带贴图的猫模型 glb**（本文 `cowcat.glb`）—— 合并时贴图从它来。

> 命名约定：本文里
> - `TEEsavR23oF.basis.glb` = 远端当前场景（已含第一只猫 `catv3_*`，**有贴图**）
> - `twocat.glb` = Blender 导出的两只猫布局（**贴图坏**，仅给位姿）
> - `cowcat.glb` = 新奶牛猫模型（**有贴图**，1 张 jpg）

---

## 3. 本地：清洗猫 glb + 给 layout 改名（**最大的坑在这**）

直接拿 Blender 导出的 `cowcat.glb` 当 `--cat` 喂 merge 会**直接报错**，两个原因：

1. **混入了房屋碎片节点**：`cowcat.glb` 的 scene roots 里除了猫子树，还夹着 9 个 `chunk012_…objectceiling…` / `window081_…` 这种房屋 mesh。merge 的护栏
   `node names appear in both --scene and --cat` 会立刻拦下。
2. **`norm_name` 同名碰撞**：猫节点叫 `Object_2.001`，`norm_name` 会把 `.001` 去掉变成 `Object_2`，
   而 layout 里第一只猫也有 `Object_2` → 匹配到 2 个 → merge 跳过真正的猫 → `no cat node matched`。

**解法（两处极小的 JSON-chunk 改动，不动几何）**：用 `_mk_clean_glb.py`（见本目录同款脚本思路）：

- `clean_cowcat.glb` ← 从 `cowcat.glb`：把 `scenes[0].nodes` 改成只留猫子树根（本例 `[11]`，
  即 `Sketchfab_model.001 → cat.obj… → Object_2.001`）；那 9 个房屋碎片节点变成**不可达孤儿**，
  merge 会自动忽略（既不进场景图也不进 AABB）。再把猫 mesh 节点 `Object_2.001` 改名成**唯一名** `cowcatbody`。
- `twocat_layout.glb` ← 从 `twocat.glb`：把那只奶牛猫的 mesh 节点（本例 node 228 `Object_2.001`）也改名 `cowcatbody`。

改完校验：两个文件里 `cowcatbody` 都唯一，且 layout 里第一只猫的 `Object_2` 仍在（不碰撞）。

> 关键认知：merge 的对齐用**房屋节点**（layout↔scene 同名唯一），猫的位姿用**猫节点**（cat↔layout 同名唯一）。
> 所以"让猫节点有个两边都唯一、且不与第一只猫碰撞的名字"是成败核心。

GLB 编辑要点（`_mk_clean_glb.py`）：只重写 JSON chunk（改 `scenes`/`nodes[].name`），BIN chunk 原样拷贝，
JSON 用空格补齐到 4 字节边界，重算总长度。**别动 BIN**。

---

## 4. 上传远端临时夹 + md5 校验 + 备份（别直接覆盖）

```bash
# 本地
cd D:\fuxian\test\00800-TEEsavR23oF
ssh train-server 'mkdir -p /data/jinsong.yuan/vlfm-demo/vlfm/data/scene_datasets/hm3d/val/00800-TEEsavR23oF/_incoming'
scp clean_cowcat.glb twocat_layout.glb \
  train-server:/data/jinsong.yuan/vlfm-demo/vlfm/data/scene_datasets/hm3d/val/00800-TEEsavR23oF/_incoming/

# 校验上传一致（本地 Get-FileHash -Algorithm MD5 对比远端 md5sum）
ssh train-server 'cd .../00800-TEEsavR23oF && md5sum _incoming/clean_cowcat.glb _incoming/twocat_layout.glb'

# 备份当前场景（带 md5，回滚就一条 cp）
ssh train-server 'cd .../00800-TEEsavR23oF && BK=backup_catv3_YYYYMMDD && mkdir -p $BK \
  && cp -n TEEsavR23oF.basis.glb $BK/ && cp -n TEEsavR23oF.basis.navmesh $BK/ \
  && md5sum TEEsavR23oF.basis.glb | tee $BK/basis.glb.md5'
```

> Windows PowerShell 坑：`$(date +%Y%m%d)` 会被 PowerShell 当 `Get-Date` 解析报错 → ssh 命令里**用固定日期串**或外层用单引号。

---

## 5. 远端：合并（贴图修复的本质一步）

```bash
ssh train-server
cd /data/jinsong.yuan/vlfm-demo/vlfm
source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh && conda activate vlfm_pip
SD=data/scene_datasets/hm3d/val/00800-TEEsavR23oF
python scripts/cat_demo/merge_cat_into_scene.py \
  --scene  $SD/TEEsavR23oF.basis.glb \          # 有贴图的底场景(本例已含第一只猫=双猫)
  --cat    $SD/_incoming/clean_cowcat.glb \      # 有贴图的干净猫
  --layout $SD/_incoming/twocat_layout.glb \     # 坏贴图的布局(只给位姿)
  --out    $SD/_incoming/TEEsavR23oF.cowcat.glb \
  --prefix cowcat
```

成功标志（脚本自检全过才会写文件）：
- `[align] matched N house reference node(s); alignment consistent`
- `[cat] matched 1 cat node(s); transform consistent`
- `[OK] all 45 scene images SHA256-identical; output has 46 images (+1 from cat)`
- `cat position matches Blender layout (deviation ~1e-7 m)`

> 单猫 vs 双猫：`--scene` 用**当前 basis（已含第一只猫）** = 双猫场景（有干扰猫，能演示 reject）；
> 想单猫就把 `--scene` 换成**原始无猫 basis**（本地 `TEEsavR23oF.basis.glb` 30MB 那个）。

---

## 6. 重命名覆盖

```bash
ssh train-server 'cd .../00800-TEEsavR23oF && mv _incoming/TEEsavR23oF.cowcat.glb TEEsavR23oF.basis.glb && md5sum TEEsavR23oF.basis.glb'
```

> navmesh **不用重烤**：猫在家具上不可走，房屋 navmesh 不变。

---

## 7. 重建 episode（不重跑就不会变！）

起点定型在 `…/data/datasets/objectnav/hm3d/v1/cat_demo/content/TEEsavR23oF.json.gz` 里，eval 跑多少次都一样。
**habitat 只读 `.json.gz`**，目录里裸 `.json` 是调试残留，别被它误导。

```bash
ssh train-server
cd /data/jinsong.yuan/vlfm-demo/vlfm && conda activate vlfm_pip
# 固定起点（最常用：从旧备份里挖出的客厅起点）
python scripts/cat_demo/build_cat_episode.py --prefix cowcat \
  --start -8.4331 0.1634 -2.3768 --start-yaw 118.29
# 或：不带 --start → 自动在猫附近 2-3m 选个能看见猫的可走点，朝向猫
```

它会：按 `--prefix` 找猫 mesn 节点 → 算 AABB 中心 → snap 到 navmesh → 选 view_points → 写 episode。
**目标完全由 `--prefix` 决定**：`--prefix cowcat` → 目标=奶牛猫；第一只猫(`catv3_*`)被忽略当干扰。

> 旧起点数值哪来的：从备份 `backups/catv2_*/content/TEEsavR23oF.json.gz` 里读 `start_position` 和 yaw。

---

## 8. 启动 VLM 服务 + 二次确认验证器（**key/base_url 坑**）

```bash
# 4 个基础服务(GDINO/SigLIP2/SAM/YOLO26-TRT) 固定到某张空卡, full 形态支持任意开放词 prompt
ssh train-server 'cd .../vlfm && SIGLIP_FORM=full bash scripts/launch_vlm_servers_jy.sh <空卡号>'

# 二次确认验证器(无需 GPU, 转发 crop 到云端)。单独起, 显式带 key + 正确 base_url:
ssh train-server 'cd .../vlfm && source .../conda.sh && conda activate vlfm_pip && export PYTHONPATH=$(pwd) \
  && export BAILIAN_API_KEY=sk-xxxx \
  && export BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
  && export ATTR_VERIFIER_PORT=12186 \
  && nohup python -um vlfm.vlm.attribute_verifier --port 12186 > outputs/attr_verifier.log 2>&1 &'
```

坑：
- **base_url 必须匹配 key**。标准 DashScope key（`sk-…`）配 `https://dashscope.aliyuncs.com/compatible-mode/v1`；
  若配成别的 MaaS 端点（如 `token-plan…maas`）会 **HTTP 401 invalid_api_key**，verify 退化成本地启发式色比判据。
- 端口若被旧验证器占用 → 新进程 `Address already in use`。先 `ss -ltnp | grep 12186` 找 PID，`kill` 掉再起。
- 验证器进程要把 key 放进**它自己的环境**（不是 eval 进程）；eval 进程只往 `localhost:12186` POST crop。
- 健康探针：`curl http://127.0.0.1:<port>/` 返回 **404 即正常**（Flask 活着，真实路由是 `/gdino`、`/verify` 等；返回空/000 才是挂了）。

---

## 9. 跑（用 runner 脚本，避开 PowerShell 引号地狱）

带空格的 `VLFM_ATTR_PREDICATE="a black-and-white cow-patterned cat"` 经 PowerShell→ssh 传会被吃掉引号
→ `black-and-white: command not found`。**解法：把环境写进一个 runner 脚本传上去跑**（字节精确，无引号问题）：

`scripts/cat_demo/_run_cowcat.sh`：
```bash
#!/usr/bin/env bash
set -u
cd /data/jinsong.yuan/vlfm-demo/vlfm
export CUDA_VISIBLE_DEVICES=2,7          # sim=cuda:0(物理2), torch=cuda:1(物理7); 选空闲卡, 别和 100% 卡撞
export VIDEO_DIR=video_dir/cowcat_demo
export LOG=outputs/cowcat_demo.log
export TB_DIR=tb/cowcat_demo
export VLFM_OBJECTNAV_QUERY=找奶牛猫
export VLFM_ATTR_NOUN=cat                # 显式给检测名词, 绕开云端解析
export VLFM_ATTR_PREDICATE="a black-and-white cow-patterned cat"  # 显式判据(离线启发式没有"奶牛/cow"别名)
export VLFM_ATTR_VERIFY=1
export VLFM_ATTR_FAIL_OPEN=0             # 验证不了就拒绝STOP(而非误放行)
export ATTR_VERIFIER_PORT=12186
exec bash scripts/cat_demo/eval_cat_demo.sh
```
```bash
# 传上去(注意去 CRLF)再跑
scp scripts\cat_demo\_run_cowcat.sh train-server:/data/jinsong.yuan/vlfm-demo/vlfm/scripts/cat_demo/_run_cowcat.sh
ssh train-server "sed -i 's/\r$//' .../scripts/cat_demo/_run_cowcat.sh && bash .../scripts/cat_demo/_run_cowcat.sh"
```

成功标志：`Average episode success: 1.0000`、`target_detected=1`、`stop_called=1`、`EXIT=0`；
二次确认走云端时日志有 `[attr] verify[bailian] match=True: …`（走启发式则是 `verify[heuristic] match=True: color_ratio=…`）。

视频在 `<VIDEO_DIR>/episode=0-...success=1.00....mp4`。

---

## 10. 坑表（速查）

| # | 坑 | 解 |
|---|---|---|
| 1 | Blender 导出后房子没贴图 | 正常；Blender 只摆位，贴图靠 merge 从原始 basis + 猫模型拷回 |
| 2 | `--cat` 混入房屋碎片 → `names in both scene and cat` | 清洗：`scenes[0].nodes` 只留猫子树根 |
| 3 | 猫节点 `Object_2.001` 与第一只猫 `Object_2` 经 norm_name 碰撞 → `no cat node matched` | 把猫 mesh 节点改成唯一名(如 `cowcatbody`)，layout 里同步改 |
| 4 | 猫在家具上, 默认参数永远 STOP 不了 | `eval_cat_demo.sh` 已固化 `pointnav_stop_radius=1.2` + `success_distance=0.5` |
| 5 | episode 不重跑就不变；裸 `.json` 是残留 | 改完资产必跑 `build_cat_episode.py`；只认 `.json.gz` |
| 6 | `找奶牛猫` 离线解析退化成 `a cat`(无 cow 色别名) | 显式 `VLFM_ATTR_NOUN=cat` + `VLFM_ATTR_PREDICATE="a black-and-white cow-patterned cat"` |
| 7 | 云端 verify 401 invalid_api_key | `BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1` 配标准 DashScope key |
| 8 | sim 与 torch 同卡 → EGL renderer 冻结 | 拆卡 `CUDA_VISIBLE_DEVICES=<simEmpty>,<torch>` |
| 9 | PowerShell 吃掉带空格/引号的 env | 用 runner 脚本(scp 传) 而非 ssh 内联 |
| 10 | 12186 端口被旧验证器占 | `ss -ltnp|grep 12186` → `kill` → 重起 |
| 11 | ssh 内联 `$(date)` 被 PowerShell 解析报错 | 固定日期串 / 外层单引号 |

---

## 附：本次(2026-06-18)实测数值

- 旧 basis md5 `a051bc448f0764a0d8465ebe62c77249`（= 本地 `TEEsavR23oF.catv3.glb`，单猫含 `catv3_*`）
- 合并后 basis md5 `05c3638805fb4557dc291472cd2072eb`（双猫；46 images）
- 奶牛猫节点：`cowcatbody`（清洗自 `cowcat.glb` 的 `Object_2.001`）
- 奶牛猫位置(Habitat) `(-0.705, 1.166, -2.662)`；高 1.17m（在家具上，故需 stop_radius 覆盖）
- 起点 `(-8.4331, 0.1634, -2.3768)`，yaw `+118.29°`，geodesic 6.44m，194 个 view_points
- 结果 `success=1.0`, `distance_to_goal=0.06m`, `spl=0.30`
- 二次确认(云端)：`verify[bailian] match=True: The cat displays a distinct black-and-white cow-like pattern…`
- 备份：`…/00800-TEEsavR23oF/backup_catv3_20260618/`
- runner：`scripts/cat_demo/_run_cowcat.sh`
