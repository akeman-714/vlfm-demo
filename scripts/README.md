# scripts/

Grouped by role. The two things you actually launch live at the root; everything
else is in a labeled folder.

## Entry points (run these)

| Script | What it does |
|---|---|
| `eval_1b_parallel.sh` | The validated **"1b" parallel** ObjectNav eval. N habitat renderers, each ALONE on its own CUDA-free GPU; all torch/nav contexts + the 4 VLM servers share `VLM_NAV_CARD` (default GPU0). Two EGL renderers can't share a card, so the GPU floor is N+1. |
| `launch_vlm_servers_jy.sh` | Start the 4 VLM Flask servers (GroundingDINO / BLIP2-ITM / SAM / YOLOv7) pinned to one GPU (default 0). **Prerequisite** for `eval_1b_parallel.sh` and everything in `cat_demo/`. |

```bash
bash scripts/launch_vlm_servers_jy.sh 0          # weights load in ~60-90s
RENDER_CARDS=1,2,4,5 VLM_NAV_CARD=0 bash scripts/eval_1b_parallel.sh
```

## cat_demo/ — the cat-finding demo

| File | Role |
|---|---|
| `eval_cat_demo.sh` | **Core.** One cat-finding episode (split-GPU: sim=cuda:0, torch=cuda:1). Every web experiment ultimately calls this, parameterized by env vars. |
| `web.py` / `web.sh` / `web_static/` | Flask front-end. `bash scripts/cat_demo/web.sh` launches it (default port 7861). |
| `eval_semantic_cat_demo.sh` | CLI path: natural-language request → label → `eval_cat_demo.sh`. |
| `semantic_goal_head.py` | NL→label resolver (Bailian / OpenAI-compatible). Imported by `web.py`; called by `eval_semantic_cat_demo.sh`. |

### The web "experiments" are all `eval_cat_demo.sh` + env switches

There is no separate script per experiment — the front-end spawns the same
`eval_cat_demo.sh` with a different env set:

| Web mode | Extra env on top of `eval_cat_demo.sh` | Module |
|---|---|---|
| Semantic query | `semantic_goal_head.py` resolves NL → split | — |
| Find cat | (none) | baseline |
| Global home 40 / 100 | `VLFM_GLOBAL_NAV=1` + A* back to (0,0) after 40 / 100 steps | 3 |
| Object memory cat | `VLFM_GLOBAL_NAV=1` + `VLFM_OBJECT_MEMORY_PATH=…/cat.json` | 2 + 3 |
| Persistent map + memory (2 passes) | runs twice: pass1 saves `cat_map.npz` + `cat.json`, pass2 loads them + A* | 1 + 2 + 3 |

Module env-var switches are documented in `docs/persistent_nav/`.

## lib/ — runtime dependency (do not remove)

`lib/vlfm_split_gpu_patch/sitecustomize.py` is auto-loaded via `PYTHONPATH` by
`eval_1b_parallel.sh` and `cat_demo/eval_cat_demo.sh`. It routes VLFM's hardcoded
`device="cuda"` literals onto the torch GPU so the EGL render card stays
CUDA-free. Moving or deleting it breaks both runs.

## upstream/ — original VLFM reference scripts (unmodified)

## archive/ — one-time / superseded

`build_cat_demo.py` (built the cat_demo dataset) and `smoke_cat_demo.py`
(dataset smoke test). Kept for reproducibility; not part of any run.
