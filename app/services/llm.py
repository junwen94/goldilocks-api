import json
import logging
import os
import re

import httpx

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
VLLM_MODEL = os.getenv("VLLM_MODEL", "").strip()
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))

logger = logging.getLogger(__name__)


async def _resolve_model(client: httpx.AsyncClient) -> str:
    if VLLM_MODEL:
        return VLLM_MODEL
    response = await client.get(f"{VLLM_BASE_URL}/models")
    response.raise_for_status()
    for model in response.json().get("data", []):
        if isinstance(model, dict) and model.get("id"):
            return model["id"]
    raise RuntimeError("Could not resolve model from vLLM server")


async def call_once(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Single non-streaming call to vLLM. Returns the full response dict."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
        model = await _resolve_model(client)
        payload: dict = {"model": model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            # Qwen3 thinking mode conflicts with tool call XML format
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        response = await client.post(f"{VLLM_BASE_URL}/chat/completions", json=payload)
        if not response.is_success:
            logger.warning("call_once %d: %s", response.status_code, response.text[:500])
        response.raise_for_status()
        data = response.json()
        _patch_tool_calls(data)
        return data


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _patch_tool_calls(data: dict) -> None:
    """vLLM 0.19.1 qwen3_xml parser bug: tool_calls is [] even when the model
    outputs <tool_call>JSON</tool_call> in the content field. Parse it ourselves."""
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        if msg.get("tool_calls"):
            continue
        content = msg.get("content") or ""
        matches = _TOOL_CALL_RE.findall(content)
        if not matches:
            continue
        tool_calls = []
        for raw in matches:
            try:
                obj = json.loads(raw)
                tool_calls.append({
                    "id": f"call-{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": obj["name"],
                        "arguments": json.dumps(obj.get("arguments", {})),
                    },
                })
            except (json.JSONDecodeError, KeyError):
                logger.warning("Could not parse <tool_call> content: %r", raw[:200])
        if tool_calls:
            msg["tool_calls"] = tool_calls
            msg["content"] = _TOOL_CALL_RE.sub("", content).strip() or None
            choice["finish_reason"] = "tool_calls"


def _filter_think(delta: str, state: dict) -> str | None:
    if "<think>" in delta:
        state["in_think"] = True
        before = delta[: delta.index("<think>")]
        return before or None
    if "</think>" in delta:
        state["in_think"] = False
        after = delta[delta.index("</think>") + len("</think>") :]
        return after or None
    if state.get("in_think"):
        return None
    if not state.get("started") and not delta.strip():
        return None
    state["started"] = True
    return delta


async def stream_chat(messages: list[dict]):
    think_state: dict = {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
            model = await _resolve_model(client)
            async with client.stream(
                "POST",
                f"{VLLM_BASE_URL}/chat/completions",
                json={"model": model, "messages": messages, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    try:
                        event = json.loads(data)
                        choices = event.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {}).get("content")
                        if not delta:
                            continue
                        filtered = _filter_think(delta, think_state)
                        if filtered is not None:
                            yield f"data: {json.dumps(filtered)}\n\n"
                    except json.JSONDecodeError:
                        continue
    except httpx.HTTPError as exc:
        msg = f"Error: could not reach the model service ({exc.__class__.__name__})."
        yield f"data: {json.dumps(msg)}\n\n"
        yield "data: [DONE]\n\n"
        return

    yield "data: [DONE]\n\n"
