# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Attribute/relationship verifier for two-stage ObjectNav.

The navigation policy talks to ``AttributeVerifierClient.verify(crop, predicate)``.
The default server backend calls an OpenAI-compatible vision chat endpoint (Bailian
today, self-hosted omni model later) and returns a small JSON verdict.  The policy
can also use the local heuristic fallback when the server/API is unavailable.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import requests
from PIL import Image

from .server_wrapper import host_model, image_to_str, str_to_image

DEFAULT_BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_PARSE_MODEL = "qwen3.6-flash"
DEFAULT_VERIFY_MODEL = "qwen-vl-plus"


@dataclass(frozen=True)
class ParsedInstruction:
    noun: str
    predicate: str
    original: str
    reason: str = ""


@dataclass(frozen=True)
class VerifyResult:
    match: bool
    reason: str
    source: str = "unknown"

    def to_json(self) -> dict[str, Any]:
        return {"match": bool(self.match), "reason": self.reason, "source": self.source}


def _env_first(names: Iterable[str]) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object, got {type(parsed).__name__}")
    return parsed


def _post_chat_completion(
    *,
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str,
    timeout: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        _chat_completions_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"chat completion failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"chat completion failed: {exc}") from exc

    data = json.loads(raw)
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected chat completion response: {raw}") from exc


_NOUN_ALIASES = {
    "cat": ("cat", "猫", "小猫", "猫猫", "咪咪", "喵", "喵喵"),
    "dog": ("dog", "狗", "小狗", "狗狗"),
    "chair": ("chair", "椅子", "座椅"),
    "cup": ("cup", "杯", "杯子", "水杯"),
    "bottle": ("bottle", "瓶", "瓶子"),
    "toilet": ("toilet", "马桶", "厕所", "坐便器"),
    "bed": ("bed", "床"),
    "couch": ("couch", "sofa", "沙发"),
    "potted plant": ("potted plant", "plant", "植物", "盆栽"),
}

_COLOR_ALIASES = {
    "yellow": ("yellow", "黄色", "黄", "橘黄", "橙黄", "金色"),
    "orange": ("orange", "橙色", "橙", "橘色", "橘"),
    "red": ("red", "红色", "红"),
    "blue": ("blue", "蓝色", "蓝"),
    "green": ("green", "绿色", "绿"),
    "white": ("white", "白色", "白"),
    "black": ("black", "黑色", "黑"),
    "brown": ("brown", "棕色", "棕", "褐色"),
}


def _find_alias(text: str, aliases: dict[str, tuple[str, ...]]) -> Optional[str]:
    lower = text.lower()
    for canonical, names in aliases.items():
        if any(name.lower() in lower for name in names):
            return canonical
    return None


def heuristic_parse_instruction(text: str, default_noun: str = "") -> ParsedInstruction:
    noun = _find_alias(text, _NOUN_ALIASES) or default_noun.strip().lower()
    color = _find_alias(text, _COLOR_ALIASES)
    if noun and color:
        predicate = f"a {color} {noun}"
    elif noun:
        predicate = f"a {noun}"
    else:
        predicate = text.strip()
    return ParsedInstruction(noun=noun, predicate=predicate, original=text, reason="heuristic")


def parse_objectnav_instruction(text: str, default_noun: str = "", timeout: float = 8.0) -> ParsedInstruction:
    """Parse a natural-language ObjectNav request into detection noun + predicate.

    The cloud parse is best-effort.  On missing keys, timeouts, malformed JSON, or
    unknown nouns, we fall back to a tiny deterministic parser so tests and offline
    demos remain runnable.
    """
    text = text.strip()
    if not text:
        return ParsedInstruction(noun=default_noun, predicate="", original="", reason="empty")

    fallback = heuristic_parse_instruction(text, default_noun)
    api_key = _env_first(("BAILIAN_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"))
    if not api_key or os.environ.get("VLFM_ATTR_PARSE_CLOUD", "1") == "0":
        return fallback

    base_url = os.environ.get("BAILIAN_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("BAILIAN_PARSE_MODEL", os.environ.get("BAILIAN_MODEL", DEFAULT_PARSE_MODEL))
    allowed = ", ".join(sorted(_NOUN_ALIASES))
    messages = [
        {
            "role": "system",
            "content": (
                "You parse robot ObjectNav instructions. Return only JSON. "
                f"The noun must be one of: {allowed}. "
                "The predicate should be an English visual yes/no phrase for the target instance, "
                "including attributes or spatial relations when present. "
                'Schema: {"noun": string, "predicate": string, "reason": string}'
            ),
        },
        {"role": "user", "content": text},
    ]
    try:
        content = _post_chat_completion(
            messages=messages,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_tokens=120,
        )
        parsed = _extract_json_object(content)
        noun = str(parsed.get("noun", "")).strip().lower()
        predicate = str(parsed.get("predicate", "")).strip()
        reason = str(parsed.get("reason", "")).strip()
        if noun not in _NOUN_ALIASES:
            return fallback
        if not predicate:
            predicate = fallback.predicate or f"a {noun}"
        return ParsedInstruction(noun=noun, predicate=predicate, original=text, reason=reason or "cloud")
    except Exception as exc:
        print(f"[attr] instruction parse fallback: {exc}", flush=True)
        return fallback


def _encode_jpeg_b64_rgb(image: np.ndarray, quality: int = 90) -> str:
    image = np.asarray(image, dtype=np.uint8)
    with io.BytesIO() as buf:
        Image.fromarray(image).convert("RGB").save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _recognized_colors(predicate: str) -> set[str]:
    lower = predicate.lower()
    colors = set()
    for color, aliases in _COLOR_ALIASES.items():
        if any(alias.lower() in lower for alias in aliases):
            colors.add(color)
    return colors


def heuristic_verify(crop_rgb: np.ndarray, predicate: str) -> VerifyResult:
    """Small local fallback for simple color predicates.

    This is intentionally conservative: if it cannot reason about the predicate, it
    returns match=True so a network outage does not block ordinary ObjectNav.
    """
    colors = _recognized_colors(predicate)
    if not colors:
        return VerifyResult(True, "no local attribute rule; fail-open", "heuristic")

    if crop_rgb.size == 0:
        return VerifyResult(False, "empty crop", "heuristic")

    hsv = cv2.cvtColor(np.asarray(crop_rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    valid = (s > 35) & (v > 35)
    if not np.any(valid):
        return VerifyResult(False, "crop has too little colored area", "heuristic")

    masks = []
    if "yellow" in colors:
        masks.append(((h >= 18) & (h <= 42) & (s > 45) & (v > 45)) | ((h >= 10) & (h <= 25) & (s > 55)))
    if "orange" in colors:
        masks.append((h >= 5) & (h <= 25) & (s > 45) & (v > 45))
    if "red" in colors:
        masks.append(((h <= 8) | (h >= 170)) & (s > 50) & (v > 45))
    if "blue" in colors:
        masks.append((h >= 90) & (h <= 130) & (s > 45) & (v > 45))
    if "green" in colors:
        masks.append((h >= 35) & (h <= 85) & (s > 45) & (v > 45))
    if "white" in colors:
        masks.append((s < 45) & (v > 150))
    if "black" in colors:
        masks.append(v < 55)
    if "brown" in colors:
        masks.append((h >= 5) & (h <= 25) & (s > 40) & (v >= 35) & (v <= 180))

    if not masks:
        return VerifyResult(True, "recognized predicate but no local rule; fail-open", "heuristic")

    color_mask = np.logical_or.reduce(masks) & valid
    ratio = float(color_mask.sum() / max(1, valid.sum()))
    threshold = float(os.environ.get("VLFM_ATTR_HEURISTIC_COLOR_RATIO", "0.035"))
    match = ratio >= threshold
    return VerifyResult(match, f"color_ratio={ratio:.3f} threshold={threshold:.3f}", "heuristic")


def _cloud_yellow_family_accept(predicate: str, reason: str) -> bool:
    lower_predicate = predicate.lower()
    if "yellow" not in lower_predicate and "黄色" not in predicate:
        return False

    lower_reason = reason.lower()
    if "cat" not in lower_reason:
        return False
    if any(token in lower_reason for token in ("not a cat", "non-cat", "no cat", "no visible")):
        return False
    if any(token in lower_reason for token in ("gray", "grey", "blue", "black", "white", "dark-brown", "dark brown")):
        return False

    accepted = ("golden", "ginger", "orange", "tan", "light-brown", "light brown", "yellowish")
    return any(token in lower_reason for token in accepted)


class AttributeVerifierClient:
    def __init__(self, port: int = 12186, host: str = "localhost") -> None:
        self.url = f"http://{host}:{port}/verify"

    def verify(self, crop_rgb: np.ndarray, predicate: str, timeout: float = 3.0) -> Optional[dict[str, Any]]:
        crop_bgr = cv2.cvtColor(np.asarray(crop_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        payload = {
            "image": image_to_str(crop_bgr),
            "predicate": predicate,
        }
        try:
            resp = requests.post(self.url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if "match" not in data:
                return None
            return {
                "match": bool(data["match"]),
                "reason": str(data.get("reason", "")),
                "source": str(data.get("source", "server")),
            }
        except Exception as exc:
            print(f"[attr] verifier unavailable: {exc}", flush=True)
            return None


class AttributeVerifierServer:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or _env_first(("BAILIAN_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"))
        self.base_url = base_url or os.environ.get("BAILIAN_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("BAILIAN_VERIFY_MODEL", DEFAULT_VERIFY_MODEL)
        self.timeout = timeout if timeout is not None else float(os.environ.get("BAILIAN_VERIFY_TIMEOUT", "12"))

    def verify(self, crop_rgb: np.ndarray, predicate: str) -> VerifyResult:
        if os.environ.get("VLFM_ATTR_VERIFY_CLOUD", "1") == "0" or not self.api_key:
            return heuristic_verify(crop_rgb, predicate)

        image_b64 = _encode_jpeg_b64_rgb(crop_rgb)
        lower_predicate = predicate.lower()
        yellow_guidance = ""
        if "yellow" in lower_predicate or "黄色" in predicate:
            yellow_guidance = (
                " For this predicate, answer true if the visible target is in the "
                "yellow/golden/ginger/orange/tan/light-brown family; do not require "
                "saturated yellow in low-resolution indoor simulator crops."
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict visual verification module for a robot. "
                    "Look only at the provided crop. Return only JSON with boolean match and short reason. "
                    "For the color phrase 'yellow cat', accept yellow, golden, ginger/orange, tan, "
                    "or light-brown fur under indoor lighting; reject clearly dark-brown, black, "
                    "white, gray, or non-cat targets. "
                    'Schema: {"match": boolean, "reason": string}'
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Does this crop match the target predicate: {predicate!r}? "
                            "Answer true only if the visible object satisfies the attributes/relations."
                            f"{yellow_guidance}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ]
        content = _post_chat_completion(
            messages=messages,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_tokens=120,
        )
        data = _extract_json_object(content)
        match_raw = data.get("match", False)
        if isinstance(match_raw, str):
            match = match_raw.strip().lower() in {"true", "yes", "y", "1", "match"}
        else:
            match = bool(match_raw)
        reason = str(data.get("reason", "")).strip()
        if not match and _cloud_yellow_family_accept(predicate, reason):
            match = True
            reason = f"cloud described an accepted yellow-family cat: {reason}"
        return VerifyResult(match, reason or "cloud verdict", "bailian")

    def process_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        crop_bgr = str_to_image(payload["image"])
        crop_rgb = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        predicate = str(payload.get("predicate", "")).strip()
        if not predicate:
            return VerifyResult(True, "empty predicate; nothing to verify", "server").to_json()
        try:
            return self.verify(crop_rgb, predicate).to_json()
        except Exception as exc:
            print(f"[attr] cloud verify fallback: {exc}", flush=True)
            return heuristic_verify(crop_rgb, predicate).to_json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("ATTR_VERIFIER_PORT", "12186")))
    parser.add_argument("--base-url", default=os.environ.get("BAILIAN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("BAILIAN_VERIFY_MODEL", DEFAULT_VERIFY_MODEL))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("BAILIAN_VERIFY_TIMEOUT", "12")))
    args = parser.parse_args()

    server = AttributeVerifierServer(base_url=args.base_url, model=args.model, timeout=args.timeout)
    key_state = "present" if server.api_key else "missing"
    print(f"[attr] verifier model={server.model} api_key={key_state} base_url={server.base_url}")
    print(f"[attr] hosting on port {args.port}...")
    host_model(server, name="verify", port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
