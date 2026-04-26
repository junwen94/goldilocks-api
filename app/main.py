from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from app.routes import chat, dft, fetch_structure, mlip, structure_match
from app.services import jarvis_cache
from app.services.llm import VLLM_BASE_URL, VLLM_MODEL

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    jarvis_cache.load()
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
