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

Edge-deployment prep (env-var switches, all default-off / wire-compatible). Two
accelerated forms are supported by this service class, sharing the same vision TRT
engine:

* **Table path (edge-lean)**: vision TRT engine + COCO text *table*, no text engine.
  Skips text tower / tokenizer at request time AND drops the text engine entirely, so
  no torch model is loaded (lowest memory). Resolves text by table lookup only --
  prompts outside the table (multi-target ``a/b``, custom templates) raise; use the
  full-tower path for those. This is the artifact set a future model-free edge runtime
  can reuse. COCO-80 prompts only by default.
* **Full tower TRT path**: vision TRT engine + text TRT engine + tokenizer. This skips
  torch tower inference at request time and supports arbitrary text (optionally with a
  table as a fast path; table misses fall back to the text engine).

Switches:

* ``SIGLIP_TEXT_CACHE`` (default on): cache normalized text embeddings per prompt.
* ``SIGLIP_TEXT_TABLE`` (path): resolve text by offline-precomputed table lookup,
  skipping the text tower (basis for the edge-lean form).
* ``SIGLIP_NUMPY_PREPROC`` (default off): transformers-free numpy/PIL image transform.
* ``SIGLIP_VISION_ENGINE`` / ``SIGLIP_TEXT_ENGINE`` (paths): run that tower from a
  TensorRT engine (built by ``scripts/siglip2_trt``) instead of torch. Engines are
  bound to the build GPU's SM arch + TRT version -- rebuild on the target device.
"""

import json
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


class _TRTRunner:
    """Fixed-shape, single-input TensorRT-10 runner (cuda-python H2D/D2H).

    This is the hand-written engine binding that ultralytics provides for YOLO but a
    raw HF model must supply itself: load the engine, allocate device buffers, copy in,
    ``execute_async_v3`` (name-based I/O), copy out. ``tensorrt`` / ``cuda`` are imported
    lazily so this module stays importable on a torch-only box (engines just aren't used).
    """

    def __init__(self, engine_path: str) -> None:
        import tensorrt as trt
        from cuda import cudart

        self._trt = trt
        self._cudart = cudart
        self._logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self._logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        input_names = []
        output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)

        if len(input_names) != 1 or len(output_names) < 1:
            raise RuntimeError(
                f"expected one TRT input and at least one output, got inputs={input_names} outputs={output_names}"
            )
        self.input_name = input_names[0]

        self.in_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        if any(dim < 0 for dim in self.in_shape):
            raise RuntimeError(f"dynamic TRT input shapes are not supported: in={self.in_shape}")

        self._output_shapes = {}
        self._output_dtypes = {}
        self._output_nbytes = {}
        for name in output_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"dynamic TRT output shapes are not supported: {name}={shape}")
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            self._output_shapes[name] = shape
            self._output_dtypes[name] = dtype
            self._output_nbytes[name] = int(np.prod(shape)) * np.dtype(dtype).itemsize
        rank2_outputs = [n for n in output_names if len(self._output_shapes[n]) == 2]
        named_rank2_outputs = [
            n for n in rank2_outputs if n in {"image_embeds", "text_embeds"} or n.endswith("_embeds")
        ]
        if len(rank2_outputs) == 1:
            self.output_name = rank2_outputs[0]
        elif named_rank2_outputs:
            self.output_name = named_rank2_outputs[0]
        else:
            self.output_name = output_names[0]
        if len(output_names) > 1:
            shape_map = {n: self._output_shapes[n] for n in output_names}
            print(f"[TRTRunner] multiple outputs {shape_map}; returning {self.output_name}")
        self.out_shape = self._output_shapes[self.output_name]
        self.out_dtype = self._output_dtypes[self.output_name]

        self._in_nbytes = int(np.prod(self.in_shape)) * np.dtype(self.in_dtype).itemsize
        self._d_in = self._malloc(self._in_nbytes)
        self._d_outputs = {name: self._malloc(nbytes) for name, nbytes in self._output_nbytes.items()}
        err, self._stream = cudart.cudaStreamCreate()
        self._check(err, "cudaStreamCreate")
        self.context.set_tensor_address(self.input_name, int(self._d_in))
        for name, ptr in self._d_outputs.items():
            self.context.set_tensor_address(name, int(ptr))

    def _check(self, err: Any, what: str) -> None:
        if int(err) != 0:
            raise RuntimeError(f"{what} failed: cuda error {int(err)}")

    def _malloc(self, nbytes: int) -> int:
        err, ptr = self._cudart.cudaMalloc(nbytes)
        self._check(err, "cudaMalloc")
        return ptr

    def infer(self, x: np.ndarray) -> np.ndarray:
        """Run the engine on a single host array, returning the host output array."""
        cudart = self._cudart
        x = np.ascontiguousarray(x, dtype=self.in_dtype)
        if tuple(x.shape) != self.in_shape:
            raise ValueError(f"TRT input shape mismatch: expected {self.in_shape}, got {tuple(x.shape)}")
        out = np.empty(self.out_shape, dtype=self.out_dtype)
        self._check(
            cudart.cudaMemcpyAsync(
                self._d_in, x.ctypes.data, self._in_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream,
            )[0],
            "H2D",
        )
        if not self.context.execute_async_v3(int(self._stream)):
            raise RuntimeError("execute_async_v3 failed")
        self._check(
            cudart.cudaMemcpyAsync(
                out.ctypes.data, self._d_outputs[self.output_name], self._output_nbytes[self.output_name],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream,
            )[0],
            "D2H",
        )
        self._check(cudart.cudaStreamSynchronize(self._stream)[0], "cudaStreamSynchronize")
        return out

    def __del__(self) -> None:
        try:
            if hasattr(self, "_d_in"):
                self._cudart.cudaFree(self._d_in)
            if hasattr(self, "_d_outputs"):
                for ptr in self._d_outputs.values():
                    self._cudart.cudaFree(ptr)
            if hasattr(self, "_stream"):
                self._cudart.cudaStreamDestroy(self._stream)
        except Exception:
            pass


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

        # Processor (tokenizer + image-processor config) is CPU-only -- it costs no GPU
        # memory, so both the torch and the TRT paths always load it.
        self.processor = AutoProcessor.from_pretrained(model_id)

        # --- vision / text tower TRT engines（边缘固化；按开关启用，缺包不影响 torch 路径） ---
        # vision-engine + 查表 = edge-lean；vision-engine + text-engine + tokenizer = full。
        # 先加载引擎,因为是否需要 torch model 取决于双塔是否都已 TRT 化。
        self._vision_trt = None
        self._text_trt = None
        v_engine = os.environ.get("SIGLIP_VISION_ENGINE")
        t_engine = os.environ.get("SIGLIP_TEXT_ENGINE")
        if v_engine:
            self._vision_trt = _TRTRunner(v_engine)
            print(f"[SigLIP2ITM] vision TRT engine: {v_engine}")
        if t_engine:
            self._text_trt = _TRTRunner(t_engine)
            print(f"[SigLIP2ITM] text TRT engine: {t_engine}")

        # 固化的省显存关键:vision 已 TRT 化、且文本侧能不靠 torch 解析(text 引擎或离线查表)时,
        # image+text 都能不靠 torch model 处理,于是完全跳过 HF 权重加载(并让 cosine 走 numpy,
        # 连 torch CUDA context 都不建)。两种 TRT 文本形态:
        #   * vision + text 引擎          = full-tower,任意 prompt(查表未命中→text 引擎兜底)
        #   * vision + 查表(无 text 引擎) = edge-lean,仅表内 prompt(未命中→报错),省掉 text 引擎显存
        # 其他组合(无 vision / 文本两路都缺)仍需 torch model 兜底,照常加载。
        table_path = os.environ.get("SIGLIP_TEXT_TABLE")
        text_resolvable_trt = self._text_trt is not None or table_path is not None
        self._skip_torch_model = self._vision_trt is not None and text_resolvable_trt
        if self._skip_torch_model:
            self.model = None
            print(f"[SigLIP2ITM] TRT towers: torch model NOT loaded (edge-lean) device={device}")
        else:
            self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
            print(f"[SigLIP2ITM] loaded model={model_id} device={device} dtype={dtype}")

        # --- 文本侧：运行时缓存 + 离线预算表（为边缘去掉 text 模型铺路） ---
        # cosine() 收到的 txt 是固定模板替换后的整串 prompt（itm_policy.py:197），target
        # 取值有限，故 text embedding 可缓存 / 离线预算，运行时跳过 text tower。
        # 不加载 torch model 时,缓存/表用 numpy 存(避免为这些小向量建 torch CUDA context)。
        self._text_cache_enabled = os.environ.get("SIGLIP_TEXT_CACHE", "1") != "0"
        self._text_cache: Dict[str, torch.Tensor] = {}
        self._text_cache_np: Dict[str, np.ndarray] = {}
        self._text_table: Optional[Dict[str, torch.Tensor]] = None
        self._text_table_np: Optional[Dict[str, np.ndarray]] = None
        if table_path:
            if self._skip_torch_model:
                self._text_table_np = self._load_text_table_np(table_path)
                n_prompts = len(self._text_table_np)
            else:
                self._text_table = self._load_text_table(table_path)
                n_prompts = len(self._text_table)
            print(f"[SigLIP2ITM] text table: {n_prompts} prompts from {table_path}")

        # --- 图像预处理：可切到 transformers-free 的 numpy/PIL 复现（为边缘去依赖） ---
        # 参数取自 HF processor，与 preprocessor_config.json 一致；边缘 runtime 可改为
        # 直接从该 json 读取，从而完全不依赖 transformers。
        self._numpy_preproc = os.environ.get("SIGLIP_NUMPY_PREPROC", "0") == "1"
        ip = self.processor.image_processor
        self._pp_size = (int(ip.size["height"]), int(ip.size["width"]))
        self._pp_mean = np.array(ip.image_mean, dtype=np.float32).reshape(3, 1, 1)
        self._pp_std = np.array(ip.image_std, dtype=np.float32).reshape(3, 1, 1)
        self._pp_rescale = float(ip.rescale_factor)
        self._pp_resample = int(ip.resample)  # PIL resample enum (2 == BILINEAR for SigLIP)

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
        # edge-lean（仅查表、无 text 引擎）下任意现编 prompt 都会落空报错,
        # 用表内任一 prompt 预热,才能真正走通 vision 引擎 + 查表路径。
        if self._text_table_np is not None and self._text_trt is None:
            warm_txt = next(iter(self._text_table_np))
        else:
            warm_txt = "a photo of a warmup target."
        self.cosine(dummy, warm_txt)

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

    def _load_text_table(self, path: str) -> Dict[str, torch.Tensor]:
        """Load an offline-precomputed, L2-normalized text-embedding table.

        ``path`` is a ``.npy`` of shape ``[N, D]`` (fp16); a sibling
        ``<stem>_meta.json`` carries the prompt strings whose order matches the npy
        rows. Hitting this table lets the text tower be skipped entirely at run time
        -- the basis for dropping the text model on edge devices. The table is built
        offline by ``scripts/siglip2_trt/build_text_table.py`` and is hardware-agnostic.
        """
        meta_path = os.path.splitext(path)[0] + "_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        prompts = meta["prompts"]
        embeds = np.load(path)
        if embeds.shape[0] != len(prompts):
            raise ValueError(f"text table rows {embeds.shape[0]} != #prompts {len(prompts)}")
        t = torch.from_numpy(embeds).to(self.device, self.dtype)
        return {prompts[i]: t[i : i + 1] for i in range(len(prompts))}

    def _get_text_embeds(self, txt: str) -> torch.Tensor:
        """Return the L2-normalized text embedding for ``txt`` (shape ``[1, D]``).

        Resolution order, cheapest first: run-time cache -> offline table -> text TRT
        engine -> torch text tower (fallback). Table / engine hits skip the torch text
        tower; the table even skips the tokenizer. Table misses (e.g. multi-target
        ``a/b`` combos) fall through to the engine or torch.
        """
        if self._text_cache_enabled and txt in self._text_cache:
            return self._text_cache[txt]
        if self._text_table is not None and txt in self._text_table:
            emb = self._text_table[txt]
        elif self._text_trt is not None:
            ti = self.processor(text=[txt], padding="max_length", truncation=True, return_tensors="pt")
            raw = self._text_trt.infer(ti["input_ids"].numpy())
            emb = torch.from_numpy(raw).to(self.device, self.dtype)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        else:
            # SigLIP text tower expects fixed-length (max_length) padding.
            text_inputs = self._to_model_inputs(
                self.processor(text=[txt], padding="max_length", truncation=True, return_tensors="pt")
            )
            with torch.inference_mode():
                emb = self._feature_tensor(self.model.get_text_features(**text_inputs))
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        if self._text_cache_enabled:
            self._text_cache[txt] = emb
        return emb

    def _load_text_table_np(self, path: str) -> Dict[str, np.ndarray]:
        """numpy 版离线文本表（无 torch model 时用），返回 {prompt: 已 L2 归一化的 np[1, D]}。"""
        meta_path = os.path.splitext(path)[0] + "_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        prompts = meta["prompts"]
        embeds = np.load(path).astype(np.float32)
        if embeds.shape[0] != len(prompts):
            raise ValueError(f"text table rows {embeds.shape[0]} != #prompts {len(prompts)}")
        return {prompts[i]: embeds[i : i + 1] for i in range(len(prompts))}

    def _get_text_embeds_np(self, txt: str) -> np.ndarray:
        """L2 归一化 text embedding（numpy [1, D]），torch-model-free。

        解析顺序：运行时缓存 -> 离线表 -> text TRT 引擎。这是 TRT 文本侧，永不触碰 torch
        text tower（表命中还省掉 tokenizer）。edge-lean（无 text 引擎）下表未命中即报错。
        """
        if self._text_cache_enabled and txt in self._text_cache_np:
            return self._text_cache_np[txt]
        if self._text_table_np is not None and txt in self._text_table_np:
            emb = self._text_table_np[txt]
        elif self._text_trt is not None:
            ti = self.processor(text=[txt], padding="max_length", truncation=True, return_tensors="np")
            raw = self._text_trt.infer(np.ascontiguousarray(ti["input_ids"])).astype(np.float32)
            emb = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
        else:
            raise KeyError(
                f"prompt not in text table and no SIGLIP_TEXT_ENGINE set (edge-lean table mode): {txt!r}. "
                f"Set SIGLIP_TEXT_ENGINE (full-tower) for arbitrary/multi-target prompts, "
                f"or bake this prompt into the table offline."
            )
        if self._text_cache_enabled:
            self._text_cache_np[txt] = emb
        return emb

    def _cosine_trt(self, image: np.ndarray, txt: str) -> float:
        """Full-tower TRT cosine，全程 numpy：无 torch model、无 torch CUDA context。"""
        inputs = self._preprocess_image(image)
        pix = inputs["pixel_values"]
        pix = pix.cpu().numpy() if torch.is_tensor(pix) else np.asarray(pix)
        img = self._vision_trt.infer(pix).astype(np.float32)
        img = img / np.linalg.norm(img, axis=-1, keepdims=True)
        txt_emb = self._get_text_embeds_np(txt)
        return float((img * txt_emb).sum(axis=-1).reshape(-1)[0])

    def _numpy_pixel_values(self, image: np.ndarray) -> torch.Tensor:
        """transformers-free SigLIP image transform: resize -> rescale -> normalize.

        Mirrors ``SiglipImageProcessor`` (preprocessor_config.json) using only PIL +
        numpy, so an edge runtime needs no transformers. Returns a fp32 ``[1, 3, H, W]``
        tensor (``_to_model_inputs`` later casts it to the model dtype).
        """
        h, w = self._pp_size
        pil = Image.fromarray(image).convert("RGB").resize((w, h), self._pp_resample)
        arr = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1)  # [3, H, W]
        arr = arr * self._pp_rescale
        arr = (arr - self._pp_mean) / self._pp_std
        return torch.from_numpy(np.ascontiguousarray(arr[None]))

    def _preprocess_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Produce ``{"pixel_values": ...}`` for the vision tower.

        ``SIGLIP_NUMPY_PREPROC=1`` -> transformers-free numpy/PIL path (edge);
        otherwise the HF ``AutoProcessor`` (default, dev box).
        """
        if self._numpy_preproc:
            return {"pixel_values": self._numpy_pixel_values(image)}
        pil_img = Image.fromarray(image).convert("RGB")
        return self.processor(images=pil_img, return_tensors="pt")

    def _image_embeds(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Raw (un-normalized) image embeddings, from the TRT engine or the torch tower."""
        if self._vision_trt is not None:
            raw = self._vision_trt.infer(inputs["pixel_values"].cpu().numpy())
            return torch.from_numpy(raw).to(self.device, self.dtype)
        model_inputs = self._to_model_inputs(inputs)
        return self._feature_tensor(self.model.get_image_features(**model_inputs))

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """Cosine similarity between the image and the text (L2-normalized embeddings).

        Args:
            image (np.ndarray): Input image (same convention as ``BLIP2ITM``: the array
                decoded by ``server_wrapper.str_to_image``, interpreted as RGB).
            txt (str): The text to compare the image to.

        Returns:
            float: Cosine similarity in [-1, 1].
        """
        if self.model is None:
            # Full-tower TRT: both towers are engines and the similarity is a tiny [1, D]
            # dot product -- do it all in numpy so no torch CUDA context is ever created.
            return self._cosine_trt(image, txt)

        inputs = self._preprocess_image(image)

        with torch.inference_mode():
            image_embeds = self._image_embeds(inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            # text 侧：缓存 / 预算表 / text engine 命中则跳过 torch text tower。
            text_embeds = self._get_text_embeds(txt)
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
