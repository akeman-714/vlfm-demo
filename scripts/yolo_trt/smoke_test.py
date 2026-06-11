#!/usr/bin/env python
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""G2.1 detection-level sanity check for the YOLO26 + TensorRT service.

Round-trips an image through the *running* detector (port 12184, route
``/yolov7``) using the unchanged :class:`vlfm.vlm.yolov7.YOLOv7Client`, prints
the detections, and writes an annotated frame. This exercises the full contract
path (client JPEG-encode -> server decode -> predict -> JSON -> from_json ->
annotate), making RGB/BGR, normalization, and threshold glue bugs visible.

Run in the vlfm_pip env (needs the vlfm package + torch); the server itself runs
in yolo_trt:

  PYTHONPATH=. python scripts/yolo_trt/smoke_test.py \\
      --image outputs/module3_frame_probe/t12.png --out yolo_trt_smoke.jpg
"""

import argparse

import cv2

from vlfm.vlm.yolov7 import YOLOv7Client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to an RGB-content image file")
    parser.add_argument("--port", type=int, default=12184)
    parser.add_argument("--out", default="yolo_trt_smoke.jpg")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.0,
        help="optional client-side confidence filter (policy uses 0.8 for COCO)",
    )
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"could not read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # VLFM passes RGB to the detector

    det = YOLOv7Client(port=args.port).predict(rgb)
    if args.conf > 0:
        det.filter_by_conf(args.conf)

    print(f"image={args.image}  shape={rgb.shape}  num_detections={det.num_detections}")
    print(det)

    cv2.imwrite(args.out, cv2.cvtColor(det.annotated_frame, cv2.COLOR_RGB2BGR))
    print(f"wrote annotated frame -> {args.out}")


if __name__ == "__main__":
    main()
