from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from goldilocks_core.kpoints.kmesh import k_distance_to_mesh, mesh_to_n_reduced_kpoints
from goldilocks_core.pseudo.pp_policy import PseudoPolicy
from goldilocks_core.pseudo.pp_registry import load_pseudo_metadata
from goldilocks_core.pseudo.pp_selector import select_pp_candidates_for_structure
from goldilocks_core.structure.io import load_structure

router = APIRouter()

_PSEUDO_ROOT = os.getenv("PSEUDO_ROOT", "")

_KDISTANCE_MAP: dict[str, float] = {
    "light": 0.40,
    "standard": 0.25,
    "dense": 0.15,
    "very-dense": 0.08,
}


def _parse_structure(content: str, name: str):
    suffix = Path(name).suffix or ".cif"
    with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        return load_structure(tmp_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── /api/dft/kpoints ──────────────────────────────────────────────────────────

class KpointsRequest(BaseModel):
    structure_content: str
    structure_name: str
    kdistance_id: str


class KpointsResponse(BaseModel):
    grid: tuple[int, int, int]
    n_reduced_kpoints: int
    k_distance: float
    summary: str


@router.post("/dft/kpoints", response_model=KpointsResponse)
async def dft_kpoints(req: KpointsRequest):
    if req.kdistance_id == "gamma":
        return KpointsResponse(
            grid=(1, 1, 1),
            n_reduced_kpoints=1,
            k_distance=0.0,
            summary="Γ-point only (1×1×1). Suitable only for large supercells or molecules.",
        )

    k_distance = _KDISTANCE_MAP.get(req.kdistance_id)
    if k_distance is None:
        raise HTTPException(400, f"Unknown kdistance_id: {req.kdistance_id!r}")

    structure = _parse_structure(req.structure_content, req.structure_name)
    grid = k_distance_to_mesh(structure, k_distance)
    n_reduced = mesh_to_n_reduced_kpoints(structure, grid)

    summary = (
        f"K-mesh: {grid[0]}×{grid[1]}×{grid[2]} "
        f"(k-spacing ≈ {k_distance:.2f} Å⁻¹, {n_reduced} irreducible k-points)."
    )
    return KpointsResponse(
        grid=grid,
        n_reduced_kpoints=n_reduced,
        k_distance=k_distance,
        summary=summary,
    )


# ── /api/dft/pseudo ───────────────────────────────────────────────────────────

class PseudoRequest(BaseModel):
    structure_content: str
    structure_name: str
    functional: str
    pseudo_type: str | None = None


class PseudoEntry(BaseModel):
    filename: str
    library: str | None
    functional: str | None
    pseudo_type: str | None


class PseudoResponse(BaseModel):
    candidates: dict[str, list[PseudoEntry]]
    summary: str


@router.post("/dft/pseudo", response_model=PseudoResponse)
async def dft_pseudo(req: PseudoRequest):
    if not _PSEUDO_ROOT:
        raise HTTPException(503, "PSEUDO_ROOT is not configured in the server environment.")

    structure = _parse_structure(req.structure_content, req.structure_name)

    try:
        metadata_list = load_pseudo_metadata(_PSEUDO_ROOT)
    except Exception as exc:
        raise HTTPException(503, f"Failed to load pseudopotentials: {exc}") from exc

    policy = PseudoPolicy(
        preferred_functional=req.functional,
        allowed_pseudo_types=(req.pseudo_type,) if req.pseudo_type else (),
    )
    candidates = select_pp_candidates_for_structure(structure, metadata_list, policy)

    result: dict[str, list[PseudoEntry]] = {
        element: [
            PseudoEntry(
                filename=p.filename,
                library=p.library,
                functional=p.functional,
                pseudo_type=p.pseudo_type,
            )
            for p in pseudos
        ]
        for element, pseudos in candidates.items()
    }

    elements = list(candidates.keys())
    n_found = sum(len(v) for v in result.values())
    summary = (
        f"Found {n_found} pseudopotential candidate(s) for "
        f"{len(elements)} element(s): {', '.join(elements)}."
    )
    return PseudoResponse(candidates=result, summary=summary)
