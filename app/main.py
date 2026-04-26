import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from app.routes import chat, dft, fetch_structure, mlip, structure_match
from app.services import jarvis_cache
from app.services.llm import VLLM_BASE_URL, VLLM_MODEL, REQUEST_TIMEOUT

load_dotenv()

logger = logging.getLogger(__name__)


async def _warmup_vllm():
    """Ping vLLM at startup so the first user message hits a warm connection."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
            r = await client.get(f"{VLLM_BASE_URL}/models")
            r.raise_for_status()
        logger.info("vLLM warmup OK (%s)", VLLM_BASE_URL)
    except Exception as exc:
        logger.warning("vLLM warmup failed (will retry on first request): %r", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    jarvis_cache.load()
    asyncio.create_task(_warmup_vllm())
    yield


app = FastAPI(title="Goldilocks API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(dft.router, prefix="/api")
app.include_router(fetch_structure.router, prefix="/api")
app.include_router(structure_match.router, prefix="/api")
app.include_router(mlip.router, prefix="/api")


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "vllm_base_url": VLLM_BASE_URL,
        "vllm_model": VLLM_MODEL or "<auto>",
    })
