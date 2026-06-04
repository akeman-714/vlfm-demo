#!/usr/bin/env python3
"""Resolve a natural-language ObjectNav request to a VLFM target label.

The script calls an OpenAI-compatible chat/completions endpoint.  It is meant
to sit in front of the existing cat demo: a request such as "咪咪你在哪" becomes
the label "cat", which can then be routed to the already-built cat_demo split.

Configuration is intentionally environment-based so API keys do not end up in
the repository or shell scripts:

    export BAILIAN_API_KEY=...
    export BAILIAN_BASE_URL=https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
    export BAILIAN_MODEL=qwen3.6-flash
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Optional


DEFAULT_BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-flash"
DEFAULT_ALLOWED_LABELS = ("cat", "toilet")


@dataclass(frozen=True)
class GoalResolution:
    label: Optional[str]
    confidence: float
    reason: str


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
    """Parse a JSON object, tolerating a short wrapper around it."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError(f"model response JSON must be an object, got {type(data).__name__}")
    return data


def _normalize_resolution(data: dict[str, Any], allowed_labels: tuple[str, ...]) -> GoalResolution:
    label = data.get("label")
    if label in ("", "null", "none", "unknown"):
        label = None
    if label is not None:
        label = str(label).strip().lower()
        if label not in allowed_labels:
            label = None

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(data.get("reason", "")).strip()
    return GoalResolution(label=label, confidence=confidence, reason=reason)


def _build_messages(text: str, allowed_labels: tuple[str, ...]) -> list[dict[str, str]]:
    labels = ", ".join(allowed_labels)
    return [
        {
            "role": "system",
            "content": (
                "你是机器人 ObjectNav 的语义解析头。"
                "把用户口语指令解析成一个标准目标标签。"
                f"只能从这些标签中选择: {labels}。"
                "如果用户说的是猫、咪咪、猫猫、小猫、喵、喵喵、喵星人等猫的称呼，输出 cat。"
                "如果用户说的是马桶、厕所、坐便器、toilet、commode 等称呼，输出 toilet。"
                "如果无法确定目标，label 用 null。"
                "只输出一个 JSON 对象，不要输出解释性正文。"
                'JSON schema: {"label": string|null, "confidence": number, "reason": string}'
            ),
        },
        {
            "role": "user",
            "content": text,
        },
    ]


def resolve_goal(
    text: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    allowed_labels: tuple[str, ...],
    timeout: float,
) -> GoalResolution:
    payload = {
        "model": model,
        "messages": _build_messages(text, allowed_labels),
        "temperature": 0,
        "max_tokens": 80,
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
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    data = json.loads(raw)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected LLM response shape: {raw}") from exc

    parsed = _extract_json_object(str(content))
    return _normalize_resolution(parsed, allowed_labels)


def _parse_allowed_labels(value: str) -> tuple[str, ...]:
    labels = tuple(label.strip().lower() for label in value.split(",") if label.strip())
    if not labels:
        raise argparse.ArgumentTypeError("allowed label list cannot be empty")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text", nargs="*", help="Natural-language navigation request, e.g. 咪咪你在哪")
    parser.add_argument("--text", dest="text_opt", help="Natural-language navigation request")
    parser.add_argument(
        "--allowed-labels",
        default=",".join(DEFAULT_ALLOWED_LABELS),
        type=_parse_allowed_labels,
        help="Comma-separated labels the parser may return. Default: cat",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BAILIAN_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL. Defaults to BAILIAN_BASE_URL or the Bailian compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("BAILIAN_MODEL", DEFAULT_MODEL),
        help="Model name. Defaults to BAILIAN_MODEL or qwen3.6-flash.",
    )
    parser.add_argument("--timeout", default=float(os.environ.get("BAILIAN_TIMEOUT", "20")), type=float)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON result instead of only the label.",
    )
    args = parser.parse_args()

    text = args.text_opt if args.text_opt is not None else " ".join(args.text).strip()
    if not text:
        parser.error("provide a navigation request, e.g. --text '咪咪你在哪'")

    api_key = _env_first(("BAILIAN_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"))
    if not api_key:
        print(
            "Missing API key: set BAILIAN_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 2

    try:
        result = resolve_goal(
            text,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            allowed_labels=args.allowed_labels,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"semantic goal resolution failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "input": text,
                    "label": result.label,
                    "confidence": result.confidence,
                    "reason": result.reason,
                },
                ensure_ascii=False,
            )
        )
    else:
        if result.label is None:
            print("unknown")
            return 3
        print(result.label)

    return 0 if result.label is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
