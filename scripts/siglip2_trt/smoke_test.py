#!/usr/bin/env python
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Two-stage smoke test for the SigLIP2 TensorRT artifacts.

The repo intentionally keeps the HF/SigLIP env and the TensorRT env separate:

1. Build a torch reference in ``siglip2_itm`` (has torch + transformers):

   PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 \\
     /data/jinsong.yuan/miniconda3/envs/siglip2_itm/bin/python \\
     scripts/siglip2_trt/smoke_test.py --mode torch-ref

2. Check the TRT engines in ``yolo_trt`` (has tensorrt + cuda-python):

   PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 \\
     /data/jinsong.yuan/miniconda3/envs/yolo_trt/bin/python \\
     scripts/siglip2_trt/smoke_test.py --mode trt-check

The second step reuses ``vlfm.vlm.siglip2itm._TRTRunner`` so this validates the
same hand-written runtime binding used by the service path.
"""

import argparse
import json
import os
import time
from typing import Any, Dict

import numpy as np


DEFAULT_REF = "outputs/siglip2_trt_smoke/ref.npz"
DEFAULT_PROMPT = "Seems like there is a cat ahead."
DEFAULT_VISION_ENGINE = "data/siglip2_vision_b16_384_fp16.engine"
DEFAULT_TEXT_ENGINE = "data/siglip2_text_b16_fp16.engine"
DEFAULT_TEXT_TABLE = "data/siglip2_text_coco80_fp16.npy"


def _demo_image(height: int = 384, width: int = 384) -> np.ndarray:
    """Deterministic RGB test image with enough structure to avoid all-zero features."""
    yy, xx = np.mgrid[0:height, 0:width]
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[..., 0] = (xx * 3 + yy) % 256
    img[..., 1] = (yy * 5 + 37) % 256
    img[..., 2] = ((xx // 2 + yy // 3) * 7) % 256
    return img


def _l2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float((_l2(a) * _l2(b)).sum(axis=-1).reshape(-1)[0])


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _load_table_row(path: str, prompt: str) -> np.ndarray:
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    prompts = meta["prompts"]
    try:
        idx = prompts.index(prompt)
    except ValueError as exc:
        raise ValueError(f"prompt not found in text table metadata: {prompt!r}") from exc
    return np.load(path)[idx : idx + 1]


def torch_ref(args: argparse.Namespace) -> None:
    # Force the torch towers so the reference is independent of any shell env left over
    # from service experiments.
    os.environ["SIGLIP_TEXT_CACHE"] = "0"
    os.environ.pop("SIGLIP_TEXT_TABLE", None)
    os.environ.pop("SIGLIP_VISION_ENGINE", None)
    os.environ.pop("SIGLIP_TEXT_ENGINE", None)

    import torch

    from vlfm.vlm.siglip2itm import SigLIP2ITM

    t0 = time.perf_counter()
    itm = SigLIP2ITM(model_id=args.model_id)
    load_s = time.perf_counter() - t0

    image = _demo_image()
    pixel_inputs = itm._preprocess_image(image)
    text_inputs = itm.processor(text=[args.prompt], padding="max_length", truncation=True, return_tensors="pt")

    t1 = time.perf_counter()
    with torch.inference_mode():
        image_raw = itm._image_embeds(pixel_inputs).float().cpu().numpy()
        model_text_inputs = itm._to_model_inputs(text_inputs)
        text_raw = itm._feature_tensor(itm.model.get_text_features(**model_text_inputs)).float().cpu().numpy()
    infer_s = time.perf_counter() - t1

    score = _cosine(image_raw, text_raw)
    out_dir = os.path.dirname(args.ref)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        args.ref,
        image=image,
        pixel_values=pixel_inputs["pixel_values"].cpu().numpy(),
        input_ids=text_inputs["input_ids"].cpu().numpy(),
        image_raw=image_raw,
        text_raw=text_raw,
        image_norm=_l2(image_raw),
        text_norm=_l2(text_raw),
        cosine=np.array([score], dtype=np.float32),
        prompt=np.array(args.prompt),
    )
    print(
        f"[torch-ref] wrote {args.ref} prompt={args.prompt!r} "
        f"cosine={score:.6f} load_s={load_s:.2f} infer_s={infer_s:.3f}"
    )


def trt_check(args: argparse.Namespace) -> None:
    from vlfm.vlm.siglip2itm import _TRTRunner

    ref: Dict[str, Any] = dict(np.load(args.ref, allow_pickle=False))
    pixel_values = ref["pixel_values"]
    input_ids = ref["input_ids"]
    ref_image = ref["image_raw"]
    ref_text = ref["text_raw"]
    ref_score = float(ref["cosine"][0])
    prompt = str(ref["prompt"])

    t0 = time.perf_counter()
    vision = _TRTRunner(args.vision_engine)
    text = _TRTRunner(args.text_engine)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    trt_image = vision.infer(pixel_values)
    trt_text = text.infer(input_ids)
    infer_s = time.perf_counter() - t1

    image_embed_cos = _cosine(trt_image, ref_image)
    text_embed_cos = _cosine(trt_text, ref_text)
    trt_score = _cosine(trt_image, trt_text)
    score_diff = abs(trt_score - ref_score)
    image_max_abs = _max_abs(trt_image, ref_image)
    text_max_abs = _max_abs(trt_text, ref_text)

    print(f"[trt-check] prompt={prompt!r}")
    print(f"[trt-check] engines loaded in {load_s:.2f}s; infer_s={infer_s:.3f}")
    print(
        "[trt-check] "
        f"image_shape={trt_image.shape} dtype={trt_image.dtype} "
        f"embed_cos={image_embed_cos:.6f} max_abs={image_max_abs:.5f}"
    )
    print(
        "[trt-check] "
        f"text_shape={trt_text.shape} dtype={trt_text.dtype} "
        f"embed_cos={text_embed_cos:.6f} max_abs={text_max_abs:.5f}"
    )
    print(
        "[trt-check] "
        f"torch_cosine={ref_score:.6f} trt_cosine={trt_score:.6f} "
        f"abs_diff={score_diff:.6f}"
    )

    failures = []
    if image_embed_cos < args.min_embed_cos:
        failures.append(f"image embed cosine {image_embed_cos:.6f} < {args.min_embed_cos}")
    if text_embed_cos < args.min_embed_cos:
        failures.append(f"text embed cosine {text_embed_cos:.6f} < {args.min_embed_cos}")
    if score_diff > args.max_score_diff:
        failures.append(f"score diff {score_diff:.6f} > {args.max_score_diff}")

    if args.text_table:
        table_text = _load_table_row(args.text_table, prompt)
        table_cos = _cosine(table_text, ref["text_norm"])
        table_score = _cosine(trt_image, table_text)
        table_score_diff = abs(table_score - ref_score)
        print(
            "[trt-check] "
            f"table_text_cos={table_cos:.6f} table_score={table_score:.6f} "
            f"table_score_abs_diff={table_score_diff:.6f}"
        )
        if table_cos < args.min_table_cos:
            failures.append(f"text table cosine {table_cos:.6f} < {args.min_table_cos}")
        if table_score_diff > args.max_score_diff:
            failures.append(f"table score diff {table_score_diff:.6f} > {args.max_score_diff}")

    if failures:
        raise SystemExit("[trt-check] FAILED: " + "; ".join(failures))
    print("[trt-check] OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["torch-ref", "trt-check"], required=True)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model-id", default=None, help="HF model id/path; defaults to $SIGLIP_MODEL_ID")
    parser.add_argument("--vision-engine", default=DEFAULT_VISION_ENGINE)
    parser.add_argument("--text-engine", default=DEFAULT_TEXT_ENGINE)
    parser.add_argument("--text-table", default=DEFAULT_TEXT_TABLE, help="optional COCO text table check; empty disables")
    parser.add_argument("--min-embed-cos", type=float, default=0.995)
    parser.add_argument("--min-table-cos", type=float, default=0.999)
    parser.add_argument("--max-score-diff", type=float, default=0.03)
    args = parser.parse_args()

    if args.mode == "torch-ref":
        torch_ref(args)
    else:
        if args.text_table == "":
            args.text_table = None
        trt_check(args)


if __name__ == "__main__":
    main()
