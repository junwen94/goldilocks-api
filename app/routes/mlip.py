"""MLIP calculation proxy — forwards requests to janus-api."""

from __future__ import annotations

import hashlib
import logging
import os

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

JANUS_API_URL = os.getenv("JANUS_API_URL", "http://localhost:8001").rstrip("/")
_TIMEOUT = 600.0

# Map frontend model ids to janus-core arch strings
_ARCH_MAP = {
    "mace-mp": "mace_mp",
    "mace_mp": "mace_mp",
    "chgnet": "chgnet",
    "alignn": "alignn",
    "m3gnet": "m3gnet",
}


async def _upload(client: httpx.AsyncClient, content: str, filename: str) -> None:
    """Upload a structure file to janus-api DATA_DIR."""
    data = content.encode()
    md5 = hashlib.md5(data).hexdigest()
    try:
        resp = await client.post(
            f"{JANUS_API_URL}/upload/single",
            files={"file": (filename, data, "text/plain")},
            data={"file_hash": md5},
        )
    except httpx.TransportError as e:
        raise HTTPException(502, f"Cannot reach janus-api at {JANUS_API_URL} — is the SSH tunnel running? ({e})")
    if not resp.is_success:
        raise HTTPException(502, f"janus-api upload failed: {resp.text[:200]}")


async def _call(client: httpx.AsyncClient, path: str, body: dict) -> dict:
    """POST to a janus-api calculation endpoint and return the response."""
    try:
        resp = await client.post(f"{JANUS_API_URL}{path}", json=body)
    except httpx.TransportError as e:
        raise HTTPException(502, f"Cannot reach janus-api at {JANUS_API_URL} — is the SSH tunnel running? ({e})")
    if not resp.is_success:
        logger.warning("janus-api %s failed %d: %s", path, resp.status_code, resp.text[:300])
        raise HTTPException(502, f"janus-api error: {resp.text[:200]}")
    return resp.json()


# ── Request models ─────────────────────────────────────────────────────────────

class SinglepointRequest(BaseModel):
    structure_content: str
    structure_name: str
    arch: str = "mace_mp"
    properties: list[str] | None = None


class GeomOptRequest(BaseModel):
    structure_content: str
    structure_name: str
    arch: str = "mace_mp"
    fmax: float = 0.1
    steps: int = 1000
    relax_mode: str = "ionic"


class PhononsRequest(BaseModel):
    structure_content: str
    structure_name: str
    arch: str = "mace_mp"
    supercell: int = 2
    displacement: float = 0.01
    symmetrize: bool = False
    temp_min: float = 0.0
    temp_max: float = 1000.0
    temp_step: float = 50.0


class EoSRequest(BaseModel):
    structure_content: str
    structure_name: str
    arch: str = "mace_mp"
    min_volume: float = 0.95
    max_volume: float = 1.05
    n_volumes: int = 7
    eos_type: str = "birchmurnaghan"


class NEBRequest(BaseModel):
    init_structure_content: str
    init_structure_name: str
    final_structure_content: str
    final_structure_name: str
    arch: str = "mace_mp"
    n_images: int = 15
    fmax: float = 0.1
    steps: int = 100
    interpolator: str = "pymatgen"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/mlip/singlepoint")
async def mlip_singlepoint(req: SinglepointRequest) -> dict:
    arch = _ARCH_MAP.get(req.arch, req.arch)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        await _upload(client, req.structure_content, req.structure_name)
        result = await _call(client, "/singlepoint/", {
            "struct": req.structure_name,
            "arch": arch,
            **({"properties": req.properties} if req.properties else {}),
        })
    raw = result.get("results", {})
    energy = raw.get("energy")
    summary = f"MLIP singlepoint ({arch}): energy {energy:.4f} eV" if energy is not None else f"MLIP singlepoint ({arch}) complete"
    return {"raw": raw, "summary": summary}


@router.post("/mlip/geomopt")
async def mlip_geomopt(req: GeomOptRequest) -> dict:
    arch = _ARCH_MAP.get(req.arch, req.arch)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        await _upload(client, req.structure_content, req.structure_name)
        result = await _call(client, "/geomopt/", {
            "struct": req.structure_name,
            "arch": arch,
            "fmax": req.fmax,
            "steps": req.steps,
            "relax_mode": req.relax_mode,
        })
    raw = result.get("results", {})
    e = raw.get("final_energy")
    f = raw.get("max_force")
    summary = f"MLIP geom opt ({arch}): final energy {e:.4f} eV, max force {f:.4f} eV/Å" if e is not None else f"MLIP geom opt ({arch}) complete"
    return {"raw": raw, "summary": summary}


@router.post("/mlip/phonons")
async def mlip_phonons(req: PhononsRequest) -> dict:
    arch = _ARCH_MAP.get(req.arch, req.arch)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        await _upload(client, req.structure_content, req.structure_name)
        result = await _call(client, "/phonons/", {
            "struct": req.structure_name,
            "arch": arch,
            "supercell": req.supercell,
            "displacement": req.displacement,
            "symmetrize": req.symmetrize,
            "temp_min": req.temp_min,
            "temp_max": req.temp_max,
            "temp_step": req.temp_step,
        })
    raw = result.get("results", {})
    temps = raw.get("temperatures") or []
    has_bands = raw.get("band_svg") is not None
    summary = (
        f"MLIP phonons ({arch}): band structure and thermal properties computed for {req.structure_name}"
        if has_bands else
        f"MLIP phonons ({arch}): thermal properties computed for {req.structure_name}"
    )
    return {"raw": raw, "summary": summary}


@router.post("/mlip/eos")
async def mlip_eos(req: EoSRequest) -> dict:
    arch = _ARCH_MAP.get(req.arch, req.arch)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        await _upload(client, req.structure_content, req.structure_name)
        result = await _call(client, "/eos/", {
            "struct": req.structure_name,
            "arch": arch,
            "min_volume": req.min_volume,
            "max_volume": req.max_volume,
            "n_volumes": req.n_volumes,
            "eos_type": req.eos_type,
        })
    raw = result.get("results", {})
    bm = raw.get("bulk_modulus")
    summary = f"MLIP EoS ({arch}): bulk modulus {bm:.1f} GPa" if bm is not None else f"MLIP EoS ({arch}) complete"
    return {"raw": raw, "summary": summary}


@router.post("/mlip/neb")
async def mlip_neb(req: NEBRequest) -> dict:
    arch = _ARCH_MAP.get(req.arch, req.arch)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        await _upload(client, req.init_structure_content, req.init_structure_name)
        await _upload(client, req.final_structure_content, req.final_structure_name)
        result = await _call(client, "/neb/", {
            "init_struct": req.init_structure_name,
            "final_struct": req.final_structure_name,
            "arch": arch,
            "n_images": req.n_images,
            "fmax": req.fmax,
            "steps": req.steps,
            "interpolator": req.interpolator,
        })
    raw = result.get("results", {})
    barrier = raw.get("barrier")
    summary = f"MLIP NEB ({arch}): barrier {barrier:.3f} eV" if barrier is not None else f"MLIP NEB ({arch}) complete"
    return {"raw": raw, "summary": summary}
