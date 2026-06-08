# Cat Demo Archive

This branch/tag is the restore point for the semantic cat-finding demo.

The archived demo data is intentionally small:

- `data/datasets/objectnav/hm3d/v1/cat_demo/cat_demo.json.gz`
- `data/datasets/objectnav/hm3d/v1/cat_demo/content/TEEsavR23oF.json.gz`
- `data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json`
- `data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb`
- `data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.navmesh`
- `data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.semantic.glb`
- `data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.semantic.txt`

The web entrypoint is:

```bash
bash scripts/cat_demo/web.sh
```

Runtime expectations:

- The `vlfm_pip` conda environment exists.
- Git LFS is installed and pulled.
- Required policy/VLM weights and VLM services are available in the local environment.
- A compatible API key is provided through the page or the environment.

Minimal restore flow:

```bash
git clone git@github.com:akeman-714/vlfm-demo.git
cd vlfm-demo
git checkout cat-demo-working-20260605
git lfs pull
bash scripts/cat_demo/web.sh
```
