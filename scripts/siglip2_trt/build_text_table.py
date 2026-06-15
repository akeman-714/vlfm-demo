#!/usr/bin/env python
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Offline-precompute a SigLIP2 text-embedding table for the fixed prompt set.

The navigation prompt is a fixed template (``base_objectnav_policy.py:738``:
``"Seems like there is a target_object ahead."``) and ``target_object`` is drawn
from a *finite* object-class vocabulary (here COCO-80). So every prompt the policy
will ever send is enumerable -- we compute their L2-normalized text embeddings once,
store them as a tiny ``[N, D]`` fp16 table, and let the runtime resolve text by
lookup (``SigLIP2ITM._get_text_embeds``) instead of running the text tower.

The table is **hardware-agnostic** (just normalized vectors): it can be copied to an
edge device as-is, where -- combined with a vision TRT engine and numpy preprocessing
-- it removes the need to load the text model / tokenizer / transformers at all.

To guarantee the stored vectors are bit-for-bit what the runtime torch fallback would
produce (so the Part A equivalence check is trivially cos=1), we drive the *same*
``SigLIP2ITM._get_text_embeds`` torch path used at serving time.

Run in the siglip2_itm env:

  PYTHONPATH=. python scripts/siglip2_trt/build_text_table.py \\
      --out data/siglip2_text_coco80_fp16.npy
"""

import argparse
import json
import os

import numpy as np

# Default prompt template -- must match base_objectnav_policy.py:738 so the generated
# prompt strings are byte-identical to what itm_policy.py:197 sends at run time.
DEFAULT_TEMPLATE = "Seems like there is a target_object ahead."

# Fixed, non-template prompts the policy also sends. When EXPLORATION_THRESH>0 the value
# map is two-channel (semexp_env/eval.py:68) and adds this verbatim explore sentence; it
# carries no target_object so it can't be template-substituted. Baking it in lets the
# edge-lean table cover BOTH single- and dual-channel navigation without a text engine.
DEFAULT_EXTRA_PROMPTS = ["There is a lot of area to explore ahead."]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/siglip2_text_coco80_fp16.npy")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="prompt template with 'target_object'")
    parser.add_argument("--model-id", default=None, help="HF model id / path; defaults to $SIGLIP_MODEL_ID")
    parser.add_argument(
        "--extra-targets",
        default=None,
        help="comma-separated extra target_object values (e.g. multi-target "
        "'refrigerator/fridge,tv/monitor') to bake in with the same template, extending "
        "edge-lean (no text engine) coverage beyond COCO-80",
    )
    parser.add_argument(
        "--extra-prompts",
        default=None,
        help="comma-separated FULL prompt strings (NOT template-substituted) to bake in "
        "verbatim, on top of the built-in explore sentence; for edge-lean coverage of "
        "non-template prompts",
    )
    args = parser.parse_args()

    # Force the torch text tower (no cache / no table) so we compute real embeddings.
    os.environ["SIGLIP_TEXT_CACHE"] = "0"
    os.environ.pop("SIGLIP_TEXT_TABLE", None)

    # Imported here so the env knobs above are set before model construction.
    from vlfm.vlm.coco_classes import COCO_CLASSES
    from vlfm.vlm.siglip2itm import SigLIP2ITM

    itm = SigLIP2ITM(model_id=args.model_id)

    classes = list(COCO_CLASSES)
    if args.extra_targets:
        for t in (s.strip() for s in args.extra_targets.split(",")):
            if t and t not in classes:
                classes.append(t)
    prompts = [args.template.replace("target_object", c) for c in classes]

    # Append fixed full prompts (built-in explore sentence + user --extra-prompts), deduped.
    extra_prompts = list(DEFAULT_EXTRA_PROMPTS)
    if args.extra_prompts:
        extra_prompts += [s.strip() for s in args.extra_prompts.split(",") if s.strip()]
    baked_extra = []
    for p in extra_prompts:
        if p not in prompts:
            prompts.append(p)
            baked_extra.append(p)

    rows = []
    for prompt in prompts:
        emb = itm._get_text_embeds(prompt)  # [1, D], L2-normalized, on device
        rows.append(emb.float().cpu().numpy()[0])  # -> [D] fp32 (lossless from fp16)
    table = np.stack(rows).astype(np.float16)  # [N, D] fp16

    out = args.out
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(out, table)

    meta = {
        "template": args.template,
        "classes": classes,
        "extra_prompts": baked_extra,
        "prompts": prompts,
        "dim": int(table.shape[1]),
        "model_id": itm.model_id,
        "dtype": "float16",
        "note": "hardware-agnostic; prompts byte-match itm_policy runtime; rows align to npy order",
    }
    meta_path = os.path.splitext(out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[build_text_table] saved table {table.shape} ({table.dtype}) -> {out}")
    print(f"[build_text_table] saved meta ({len(prompts)} prompts) -> {meta_path}")


if __name__ == "__main__":
    main()
