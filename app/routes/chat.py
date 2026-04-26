import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.llm import call_once, stream_chat
from app.services.prompt import build_system_prompt
from app.services.tools import TOOL_DEFINITIONS, execute_tool

router = APIRouter()
logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 5

_TOOL_LABELS = {
    "search_materials_databases": "Searching databases…",
    "suggest_kpoints": "Calculating k-points…",
    "suggest_pseudopotentials": "Selecting pseudopotentials…",
}


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    mode: str | None = None
    experience_level: str | None = None
    workspace_state: dict | None = None


async def _stream(req: ChatRequest):
    system_prompt = build_system_prompt(req.mode, req.experience_level, req.workspace_state)
    messages: list[dict] = [{"role": "system", "content": system_prompt}] + [
        m.model_dump() for m in req.messages
    ]

    for round_n in range(_MAX_TOOL_ROUNDS):
        try:
            resp = await call_once(messages, TOOL_DEFINITIONS)
        except Exception as exc:
            logger.warning("Tool loop call_once failed (round %d): %r", round_n, exc, exc_info=True)
            break

        choices = resp.get("choices", [])
        if not choices:
            break

        choice = choices[0]
        assistant_msg = choice.get("message", {})
        tool_calls = assistant_msg.get("tool_calls") or []

        if choice.get("finish_reason") != "tool_calls" or not tool_calls:
            break

        messages.append(assistant_msg)
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            label = _TOOL_LABELS.get(fn_name, fn_name)
            yield f"event: tool_status\ndata: {json.dumps({'label': label, 'tool': fn_name})}\n\n"
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {}
            result = await execute_tool(fn_name, fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
        logger.info("Tool loop round %d: executed %d tool(s)", round_n, len(tool_calls))

    async for chunk in stream_chat(messages):
        yield chunk


@router.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        _stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
