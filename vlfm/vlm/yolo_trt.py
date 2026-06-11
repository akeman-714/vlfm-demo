# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""TensorRT-backed YOLO26 COCO detector -- a drop-in replacement for the
YOLOv7-e6e Flask service in ``vlfm/vlm/yolov7.py``.

It hosts the exact same HTTP contract under the route name ``yolov7`` so the
existing :class:`vlfm.vlm.yolov7.YOLOv7Client` (and therefore the VLFM policy)
need not change -- only the launch command and conda env differ. See
``docs/notes_zh/09_YOLO26_TensorRT替换方案.md`` for the full rationale.

Designed to run inside the standalone ``yolo_trt`` conda env (ultralytics +
tensorrt). It imports only the lightweight server helpers + the COCO class list
from vlfm (never torch / habitat), so it does not require the simulation env.
"""

import argparse
from typing import Dict, List

import cv2
import numpy as np

from vlfm.vlm.coco_classes import COCO_CLASSES

from .server_wrapper import ServerMixin, host_model, str_to_image


class YOLO26Trt:
    """YOLO26 (NMS-free, end-to-end) COCO-80 detector served from a TRT engine."""

    def __init__(
        self,
        model_path: str = "data/yolo26n.engine",
        image_size: int = 640,
        conf: float = 0.25,
    ) -> None:
        # Local import: ultralytics only exists in the yolo_trt env, and is only
        # needed server-side. Keeps the module importable elsewhere for tooling.
        from ultralytics import YOLO

        self.image_size = image_size
        self.conf = conf
        self.model = YOLO(model_path, task="detect")

        # Cache the class names once: reading ``model.names`` re-initializes the
        # TRT backend on every access (before the first inference), so read it a
        # single time and keep a plain list for predict(). Fail-fast (pit #5) if
        # it is not the COCO-80 order coco_classes.py expects, else ``phrases``
        # get silently mislabeled. Standard Ultralytics COCO weights satisfy this.
        names_map = self.model.names
        self.class_names = [names_map[i] for i in range(len(names_map))]
        if self.class_names != COCO_CLASSES:
            raise ValueError(
                "YOLO model class names do not match the COCO-80 order VLFM expects.\n"
                f"  got (n={len(self.class_names)}): {self.class_names[:5]} ...\n"
                f"  exp (n={len(COCO_CLASSES)}): {COCO_CLASSES[:5]} ..."
            )

        # Warm up the engine -- the first few inferences are slow (pit #8).
        dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        for _ in range(3):
            self.model(dummy, imgsz=self.image_size, conf=self.conf, verbose=False)

    def predict(self, image: np.ndarray) -> Dict[str, List]:
        """Detect COCO objects in an RGB image, returning the VLFM JSON contract.

        Args:
            image (np.ndarray): An RGB image (VLFM convention).

        Returns:
            dict with ``boxes`` (normalized xyxy in [0, 1]), ``logits``
            (confidence scores) and COCO ``phrases`` -- identical in shape to
            ``ObjectDetections.to_json()`` so ``YOLOv7Client`` round-trips it.
        """
        # Ultralytics treats a numpy source as BGR; VLFM passes RGB (pit #2).
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = self.model(bgr, imgsz=self.image_size, conf=self.conf, verbose=False)[0]

        h, w = image.shape[:2]
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(float)  # pixel coords
        if xyxy.shape[0] > 0:
            xyxy[:, [0, 2]] /= w  # normalize x by width
            xyxy[:, [1, 3]] /= h  # normalize y by height
            xyxy = np.clip(xyxy, 0.0, 1.0)
        cls = boxes.cls.cpu().numpy().astype(int)
        # No extra NMS: YOLO26 is already end-to-end NMS-free (pit #6).
        return {
            "boxes": xyxy.tolist(),
            "logits": boxes.conf.cpu().numpy().astype(float).tolist(),
            "phrases": [self.class_names[int(c)] for c in cls],
        }


class YOLO26TrtServer(ServerMixin, YOLO26Trt):
    def process_payload(self, payload: dict) -> dict:
        image = str_to_image(payload["image"])
        return self.predict(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12184)
    parser.add_argument("--model", type=str, default="data/yolo26n.engine")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    print(f"Loading YOLO26 TensorRT model from {args.model} ...")
    server = YOLO26TrtServer(model_path=args.model, image_size=args.imgsz, conf=args.conf)
    print("Model loaded!")
    # Route name MUST stay "yolov7" so the existing YOLOv7Client is unchanged.
    print(f"Hosting on port {args.port} (route /yolov7) ...")
    host_model(server, name="yolov7", port=args.port)
