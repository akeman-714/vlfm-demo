#!/usr/bin/env python
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Export a SigLIP2 tower (vision or text) to a fixed-shape ONNX graph.

* vision: ``pixel_values[1,3,384,384]`` -> ``image_embeds[1,D]``
* text:   ``input_ids[1,L]`` (int64)     -> ``text_embeds[1,D]``

Both towers are single-input / single-output with **fully fixed shapes** (batch=1, no
dynamic axes) -> the simplest, fastest, most stable TRT engines. The outputs are the
*raw, un-normalized* embeddings, exactly like ``SigLIP2ITM`` reads ``get_*_features``
before its own L2-normalize, so the TRT path stays numerically equivalent.

Export is done in **fp32** (the model is cast up for the trace) so onnxruntime can
verify it on CPU; the fp16 conversion happens later in ``build_engine.py`` (TRT FP16
flag). Run in the siglip2_itm env (needs torch + transformers; onnx optional, only the
ort check needs onnxruntime).

  PYTHONPATH=. python scripts/siglip2_trt/export_onnx.py --tower vision \\
      --out data/siglip2_vision_b16_384.onnx
  PYTHONPATH=. python scripts/siglip2_trt/export_onnx.py --tower text \\
      --out data/siglip2_text_b16.onnx
"""

import argparse
import os

import torch


def _feature_tensor(features: torch.Tensor) -> torch.Tensor:
    """Extract the pooled feature tensor before tracing/export."""
    if torch.is_tensor(features):
        return features
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        return features.last_hidden_state[:, 0]
    raise TypeError(f"Unsupported feature output type: {type(features)!r}")


class _VisionEmbed(torch.nn.Module):
    """Trace-friendly wrapper exposing only the image tower (get_image_features)."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.m = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return _feature_tensor(self.m.get_image_features(pixel_values=pixel_values))


class _TextEmbed(torch.nn.Module):
    """Trace-friendly wrapper exposing only the text tower (get_text_features)."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.m = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return _feature_tensor(self.m.get_text_features(input_ids=input_ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", choices=["vision", "text"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default=None, help="HF model id / path; defaults to $SIGLIP_MODEL_ID")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    # Avoid pulling a text table / cache into the export run.
    os.environ["SIGLIP_TEXT_CACHE"] = "0"
    os.environ.pop("SIGLIP_TEXT_TABLE", None)

    from vlfm.vlm.siglip2itm import SigLIP2ITM

    itm = SigLIP2ITM(model_id=args.model_id)
    model = itm.model.float()  # fp32 trace; TRT does fp16 at build time

    if args.tower == "vision":
        h, w = itm._pp_size
        wrapper = _VisionEmbed(model).eval()
        example = torch.zeros(1, 3, h, w, dtype=torch.float32, device=itm.device)
        input_names, output_names = ["pixel_values"], ["image_embeds"]
    else:
        wrapper = _TextEmbed(model).eval()
        # Build a valid fixed-length input_ids (max_length padding == the serving path).
        ti = itm.processor(text=["a photo"], padding="max_length", truncation=True, return_tensors="pt")
        example = ti["input_ids"].to(itm.device)
        input_names, output_names = ["input_ids"], ["text_embeds"]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (example,),
            args.out,
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            dynamo=False,
        )
    print(f"[export_onnx] {args.tower}: {tuple(example.shape)} -> {args.out}")

    # Optional numerical check against the torch wrapper (CPU, fp32).
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        feed = {input_names[0]: example.detach().cpu().numpy()}
        onnx_out = sess.run(None, feed)[0].astype("float32")
        with torch.inference_mode():
            ref = wrapper(example).float().cpu().numpy()
        diff = float(np.abs(onnx_out - ref).max())
        print(f"[export_onnx] onnxruntime vs torch max_abs_diff = {diff:.3e}")
    except Exception as e:  # pragma: no cover - check is best-effort
        print(f"[export_onnx] ort check skipped: {e}")


if __name__ == "__main__":
    main()
