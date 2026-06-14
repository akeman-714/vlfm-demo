# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""SigLIP2-ITM backend: a drop-in, wire-compatible replacement for BLIP2-ITM.

This module hosts a dual-tower image-text cosine-similarity service that speaks the
exact same HTTP contract as ``vlfm.vlm.blip2itm``:

* Flask route name is still ``"blip2itm"`` (so the URL path stays ``/blip2itm``).
* The request payload still carries ``image`` and ``txt``.
* The response is still ``{"response": float}``.

Because of this, the navigation stack (``itm_policy.BaseITMPolicy`` ->
``BLIP2ITMClient``) does not need to know whether BLIP2 or SigLIP2 sits behind the
port. Swapping the backend is purely a launch-time decision.

Designed to run inside an *isolated* conda env (e.g. ``siglip2_itm``) with a recent
``transformers``, so the BLIP2/LAVIS env (which pins ``transformers==4.26.0``) is
never touched.
"""

import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    from transformers import AutoModel, AutoProcessor
except ModuleNotFoundError:
    print("Could not import transformers. This is OK if you are only using the client.")

# Fixed-resolution SigLIP2 checkpoint (avoid the NaFlex variants, whose processor API
# differs). Overridable via the SIGLIP_MODEL_ID env var so the choice never leaks into
# the policy code.
DEFAULT_SIGLIP_MODEL_ID = "google/siglip2-base-patch16-384"


class SigLIP2ITM:
    """SigLIP2 dual-tower image-text matching (returns a cosine similarity)."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[Any] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if model_id is None:
            model_id = os.environ.get("SIGLIP_MODEL_ID", DEFAULT_SIGLIP_MODEL_ID)
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device(device)
        if dtype is None:
            # fp16 on CUDA to save memory (the whole point of swapping BLIP2 out);
            # fp32 on CPU for numerical safety, since CPU is only a fallback.
            dtype = torch.float16 if device.type == "cuda" else torch.float32

        self.model_id = model_id
        self.device = device
        self.dtype = dtype

        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

        print(f"[SigLIP2ITM] loaded model={model_id} device={device} dtype={dtype}")

        # Warm up so the first navigation request does not pay lazy-init / CUDA cost.
        try:
            self._warmup()
        except Exception as e:  # pragma: no cover - warmup is best-effort
            print(f"[SigLIP2ITM] warmup skipped: {e}")

    def _to_model_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensors to the model device; cast floating tensors to the model dtype.

        ``input_ids`` / ``attention_mask`` stay integer; only ``pixel_values`` (and any
        other float tensor) is cast, otherwise an fp16 model chokes on fp32 pixels.
        """
        out = {}
        for k, v in inputs.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                out[k] = v.to(self.device, self.dtype)
            elif torch.is_tensor(v):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _warmup(self) -> None:
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        self.cosine(dummy, "a photo of a warmup target.")

    @staticmethod
    def _feature_tensor(features: Any) -> torch.Tensor:
        """Extract a pooled feature tensor across transformers API variants."""
        if torch.is_tensor(features):
            return features
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
            return features.last_hidden_state[:, 0]
        raise TypeError(f"Unsupported feature output type: {type(features)!r}")

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """Cosine similarity between the image and the text (L2-normalized embeddings).

        Args:
            image (np.ndarray): Input image (same convention as ``BLIP2ITM``: the array
                decoded by ``server_wrapper.str_to_image``, interpreted as RGB).
            txt (str): The text to compare the image to.

        Returns:
            float: Cosine similarity in [-1, 1].
        """
        pil_img = Image.fromarray(image).convert("RGB")
        image_inputs = self._to_model_inputs(self.processor(images=pil_img, return_tensors="pt"))
        # SigLIP text tower expects fixed-length (max_length) padding.
        text_inputs = self._to_model_inputs(
            self.processor(text=[txt], padding="max_length", truncation=True, return_tensors="pt")
        )

        with torch.inference_mode():
            image_embeds = self._feature_tensor(self.model.get_image_features(**image_inputs))
            text_embeds = self._feature_tensor(self.model.get_text_features(**text_inputs))
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            cosine = (image_embeds * text_embeds).sum(dim=-1)

        return float(cosine.item())


class SigLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    parser.add_argument("--model-id", type=str, default=None, help="HF model id; defaults to $SIGLIP_MODEL_ID")
    args = parser.parse_args()

    print("Loading model...")

    class SigLIP2ITMServer(ServerMixin, SigLIP2ITM):
        def process_payload(self, payload: dict) -> dict:
            t0 = time.perf_counter()
            image = str_to_image(payload["image"])
            t1 = time.perf_counter()
            response = self.cosine(image, payload["txt"])
            if os.environ.get("SIGLIP_LOG_TIMINGS") == "1":
                total_ms = (time.perf_counter() - t0) * 1000.0
                decode_ms = (t1 - t0) * 1000.0
                infer_ms = total_ms - decode_ms
                print(
                    f"[SigLIP2ITM] request decode_ms={decode_ms:.1f} "
                    f"infer_ms={infer_ms:.1f} total_ms={total_ms:.1f}",
                    flush=True,
                )
            return {"response": response}

    siglip = SigLIP2ITMServer(model_id=args.model_id)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    # Route name stays "blip2itm" on purpose: the policy/client are backend-agnostic.
    host_model(siglip, name="blip2itm", port=args.port, threaded=False)
