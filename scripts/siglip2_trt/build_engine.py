#!/usr/bin/env python
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Build a fixed-shape fp16 TensorRT engine from a SigLIP2 tower ONNX.

Uses the TensorRT 10 Python ``Builder`` directly (no ``trtexec`` needed -- it is not
installed). All shapes are fixed (batch=1) so no optimization profile is required.

The resulting engine is **bound to this GPU's SM arch + this TRT version** -- it is a
dev-box artifact for equivalence testing. For edge deployment, rerun this same script
on the target device to rebuild the engine there; the ONNX (and the text table) are the
portable, hardware-agnostic artifacts.

Run in an env with tensorrt (siglip2_itm after ``pip install tensorrt-cu12``):

  PYTHONPATH=. python scripts/siglip2_trt/build_engine.py \\
      --onnx data/siglip2_vision_b16_384.onnx \\
      --engine data/siglip2_vision_b16_384_fp16.engine
"""

import argparse
import os

import tensorrt as trt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--no-fp16", action="store_true", help="build fp32 instead of fp16")
    ap.add_argument("--workspace", type=int, default=4096, help="workspace pool limit (MiB)")
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TRT 10: networks are explicit-batch; the flag is a harmless no-op kept for clarity.
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"failed to parse ONNX: {args.onnx}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace * 1024 * 1024)
    use_fp16 = (not args.no_fp16) and builder.platform_has_fast_fp16
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed (build_serialized_network returned None)")

    out_dir = os.path.dirname(args.engine)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.engine, "wb") as f:
        f.write(memoryview(serialized))  # IHostMemory supports the buffer protocol

    size_mb = os.path.getsize(args.engine) / 1e6
    print(f"[build_engine] {args.onnx} -> {args.engine} ({size_mb:.1f} MB, fp16={use_fp16})")


if __name__ == "__main__":
    main()
