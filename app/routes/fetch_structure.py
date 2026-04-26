"""Fetch a CIF file for a given database entry (MP, JARVIS, Materials Cloud, NOMAD)."""

from __future__ import annotations

import asyncio
import logging
import os
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from pymatgen.core import Lattice, Structure

from app.services import jarvis_cache

logger = logging.getLogger(__name__)
router = APIRouter()

_FETCH_TIMEOUT = 30.0


class FetchStructureResponse(BaseModel):
    filename: str
    content: str
    format: str = "cif"


@router.get("/fetch-structure", response_model=FetchStructureResponse)
async def fetch_structure(
    source: str = Query(..., description="Database name: 'Materials Project', 'JARVIS', 'Materials Cloud', or 'NOMAD'"),
    entry_id: str = Query(..., description="Entry ID as returned by /api/structure-match"),
):
    if source == "Materials Project":
        return await _fetch_mp(entry_id)
    if source == "JARVIS":
        return await asyncio.to_thread(_fetch_jarvis, entry_id)
    if source == "Materials Cloud":
        return await _fetch_mc(entry_id)
    if source == "NOMAD":
        return await _fetch_nomad(entry_id)
    raise HTTPException(400, f"Unknown source: {source!r}")


# ── Materials Project ─────────────────────────────────────────────────────────

async def _fetch_mp(entry_id: str) -> FetchStructureResponse:
    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        raise HTTPException(503, "MP_API_KEY not configured")

    def _sync():
        from mp_api.client import MPRester
        with MPRester(api_key) as mpr:
            struct = mpr.get_structure_by_material_id(entry_id)
        if struct is None:
            raise ValueError(f"No structure returned for {entry_id!r}")
        return struct.to(fmt="cif")

    try:
        cif = await asyncio.to_thread(_sync)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.warning("MP fetch failed for %s: %s", entry_id, exc)
        raise HTTPException(502, f"Failed to fetch from Materials Project: {exc}") from exc

    return FetchStructureResponse(filename=f"{entry_id}.cif", content=cif)


# ── JARVIS ────────────────────────────────────────────────────────────────────

def _fetch_jarvis(entry_id: str) -> FetchStructureResponse:
    if not jarvis_cache.is_available():
        raise HTTPException(503, "JARVIS cache not loaded")

    entry = jarvis_cache.get_by_jid(entry_id)
    if entry is None:
        raise HTTPException(404, f"JARVIS entry not found: {entry_id!r}")

    atoms = entry.get("atoms")
    if not atoms:
        raise HTTPException(404, f"No structure data for JARVIS entry {entry_id!r}")

    try:
        lattice = Lattice(atoms["lattice_mat"])
        elements = atoms["elements"]
        coords = atoms["coords"]
        cartesian = atoms.get("cartesian", False)
        struct = Structure(lattice, elements, coords, coords_are_cartesian=cartesian)
        cif = struct.to(fmt="cif")
    except Exception as exc:
        logger.warning("JARVIS CIF conversion failed for %s: %s", entry_id, exc)
        raise HTTPException(500, f"Failed to convert JARVIS structure to CIF: {exc}") from exc

    return FetchStructureResponse(filename=f"{entry_id}.cif", content=cif)


# ── Materials Cloud ───────────────────────────────────────────────────────────

async def _fetch_mc(entry_id: str) -> FetchStructureResponse:
    fields = "id,chemical_formula_reduced,lattice_vectors,species_at_sites,cartesian_site_positions"
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
        try:
            resp = await client.get(
                "https://optimade.materialscloud.org/main/mc3d-pbesol-v2/structures",
                params={
                    "filter": f'_mcloud_mc3d_id="{entry_id}"',
                    "response_fields": fields,
                },
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("MC fetch failed for %s: %s", entry_id, exc)
            raise HTTPException(502, f"Failed to fetch from Materials Cloud: {exc}") from exc

    items = resp.json().get("data", [])
    if not items:
        raise HTTPException(404, f"No Materials Cloud entry found for {entry_id!r}")

    cif = _structure_from_optimade_attrs(items[0]["attributes"], entry_id)
    return FetchStructureResponse(filename=f"{entry_id}.cif", content=cif)


# ── NOMAD ─────────────────────────────────────────────────────────────────────

async def _fetch_nomad(entry_id: str) -> FetchStructureResponse:
    fields = "lattice_vectors,species_at_sites,cartesian_site_positions,chemical_formula_reduced"
    url = f"https://nomad-lab.eu/prod/v1/optimade/structures/{quote(entry_id, safe='')}"
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
        try:
            resp = await client.get(url, params={"response_fields": fields})
            if resp.status_code == 404:
                raise HTTPException(404, f"NOMAD entry not found: {entry_id!r}")
            resp.raise_for_status()
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("NOMAD fetch failed for %s: %s", entry_id, exc)
            raise HTTPException(502, f"Failed to fetch from NOMAD: {exc}") from exc

    attrs = resp.json().get("data", {}).get("attributes", {})
    safe_id = entry_id.replace("/", "-")[:40]
    cif = _structure_from_optimade_attrs(attrs, entry_id)
    return FetchStructureResponse(filename=f"nomad-{safe_id}.cif", content=cif)


# ── Shared helper ─────────────────────────────────────────────────────────────

def _structure_from_optimade_attrs(attrs: dict, label: str) -> str:
    """Build a pymatgen Structure from OPTIMADE attributes and return CIF string."""
    try:
        lattice = Lattice(attrs["lattice_vectors"])
        species = attrs["species_at_sites"]
        positions = attrs["cartesian_site_positions"]
        struct = Structure(lattice, species, positions, coords_are_cartesian=True)
        return struct.to(fmt="cif")
    except Exception as exc:
        logger.warning("CIF conversion failed for %s: %s", label, exc)
        raise HTTPException(500, f"Failed to convert structure to CIF: {exc}") from exc
