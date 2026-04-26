"""Structure search across Materials Project, Materials Cloud, NOMAD, and JARVIS."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup

from app.services import jarvis_cache
from goldilocks_core.structure.features import extract_match_features
from goldilocks_core.structure.io import load_structure

logger = logging.getLogger(__name__)
router = APIRouter()

_OPTIMADE_TIMEOUT = 30.0
_MAX_CANDIDATES = 20


# ── Request / Response ────────────────────────────────────────────────────────

class StructureMatchRequest(BaseModel):
    mode: str  # "file" | "formula"
    structure_content: str | None = None
    structure_name: str | None = None
    formula: str | None = None


class MatchEntry(BaseModel):
    formula: str
    spacegroup: str | None
    entry_id: str | None
    source: str
    url: str
    matched: bool
    score: float | None


class StructureMatchResponse(BaseModel):
    results: list[MatchEntry]
    query_formula: str | None
    errors: dict[str, str]


# ── Helpers ───────────────────────────────────────────────────────────────────

_VASP_NAMES = {"POSCAR", "CONTCAR", "CHGCAR", "LOCPOT"}

def _parse_structure(content: str, name: str) -> Structure:
    path = Path(name)
    suffix = path.suffix
    if not suffix:
        stem = path.stem.upper()
        if any(stem.startswith(v) for v in _VASP_NAMES):
            suffix = ".vasp"
        else:
            suffix = ".cif"
    with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        return load_structure(tmp_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _optimade_to_structure(attrs: dict) -> Structure | None:
    if isinstance(attrs.get("_structure"), Structure):
        return attrs["_structure"]
    try:
        lattice = Lattice(attrs["lattice_vectors"])
        species = attrs["species_at_sites"]
        positions = attrs["cartesian_site_positions"]
        return Structure(lattice, species, positions, coords_are_cartesian=True)
    except Exception:
        return None



def _to_optimade_formula(formula: str) -> str:
    """Normalize to OPTIMADE Hill format: C first, H second, then alphabetical; no count-1 suffixes."""
    try:
        comp = Composition(formula).reduced_composition
        elements = sorted(
            comp.elements,
            key=lambda e: (e.symbol != "C", e.symbol != "H", e.symbol),
        )
        result = ""
        for el in elements:
            count = int(comp[el])
            result += el.symbol + (str(count) if count != 1 else "")
        return result
    except Exception:
        return formula


def _formula_filter(formula_reduced: str) -> str:
    normalized = _to_optimade_formula(formula_reduced)
    return f'chemical_formula_reduced = "{normalized}"'


# ── Per-database queries ──────────────────────────────────────────────────────

async def _query_optimade(
    base_url: str,
    source_name: str,
    formula_reduced: str,
    client: httpx.AsyncClient,
    url_builder,
    include_structure: bool = False,
    extra_fields: list[str] | None = None,
) -> tuple[list[dict], str | None]:
    """Query an OPTIMADE endpoint, return (raw_entries, error)."""
    filter_str = _formula_filter(formula_reduced)
    base_fields = "id,chemical_formula_reduced,elements,nsites"
    if include_structure:
        base_fields += ",lattice_vectors,species,species_at_sites,cartesian_site_positions"
    if extra_fields:
        base_fields += "," + ",".join(extra_fields)
    params = {
        "filter": filter_str,
        "response_fields": base_fields,
        "page_limit": _MAX_CANDIDATES,
    }
    try:
        resp = await client.get(f"{base_url.rstrip('/')}/structures", params=params)
        if resp.status_code == 500:
            # Server-side bug (e.g. NOMAD chokes on certain formula patterns)
            logger.warning("%s returned 500 for formula=%s — treating as empty", source_name, formula_reduced)
            return [], None
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("data", [])
        entries = []
        for item in raw:
            attrs = item.get("attributes", {})
            url = url_builder(item, attrs)
            if url:
                entries.append({"item": item, "attrs": attrs, "url": url})
        logger.info("%s: %d raw entries, %d with URL (formula=%s)", source_name, len(raw), len(entries), formula_reduced)
        return entries, None
    except Exception as exc:
        logger.warning("%s query failed: %s", source_name, exc)
        return [], str(exc)


def _mp_url(item: dict, attrs: dict) -> str | None:
    entry_id = item.get("id", "")
    if entry_id:
        return f"https://next-gen.materialsproject.org/materials/{entry_id}"
    return None


def _mc_url(item: dict, attrs: dict) -> str | None:
    mc_id = attrs.get("_mcloud_mc3d_id")
    if mc_id:
        return f"https://mc3d.materialscloud.org/details/{mc_id}/pbesol-v2"
    return None


def _nomad_url(item: dict, attrs: dict) -> str | None:
    return attrs.get("_nmd_entry_page_url") or attrs.get("_nmd_entry_id")


async def _query_mp(formula_reduced: str, client: httpx.AsyncClient):
    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        return [], "MP_API_KEY not configured"

    def _fetch():
        from mp_api.client import MPRester
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(
                formula=formula_reduced,
                fields=["material_id", "formula_pretty", "symmetry", "nsites"],
            )
        entries = []
        for doc in docs[:_MAX_CANDIDATES]:
            mp_id = str(doc.material_id)
            spg = doc.symmetry.symbol if doc.symmetry else None
            entries.append({
                "item": {"id": mp_id},
                "attrs": {
                    "chemical_formula_reduced": _to_optimade_formula(doc.formula_pretty),
                    "_computed_spacegroup": spg,
                    "nsites": getattr(doc, "nsites", None),
                },
                "url": f"https://next-gen.materialsproject.org/materials/{mp_id}",
            })
        return entries, None

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as exc:
        return [], str(exc)


async def _query_mc(formula_reduced: str, client: httpx.AsyncClient):
    return await _query_optimade(
        "https://optimade.materialscloud.org/main/mc3d-pbesol-v2",
        "Materials Cloud",
        formula_reduced,
        client,
        _mc_url,
        include_structure=True,  # always fetch structure for local spacegroup computation
        extra_fields=["_mcloud_mc3d_id"],
    )


async def _query_nomad(formula_reduced: str, client: httpx.AsyncClient):
    return await _query_optimade(
        "https://nomad-lab.eu/prod/v1/optimade",
        "NOMAD",
        formula_reduced,
        client,
        _nomad_url,
        include_structure=False,  # spacegroup available natively; structure fetch unreliable
        extra_fields=[
            "_nmd_entry_page_url",
            "_nmd_results_material_symmetry_space_group_symbol",
        ],
    )


def _spg_number_to_symbol(spg_raw) -> str | None:
    if spg_raw is None:
        return None
    try:
        return SpaceGroup.from_int_number(int(spg_raw)).symbol
    except Exception:
        return str(spg_raw)


def _enrich_with_spacegroup(entries: list[dict]) -> None:
    """Compute spacegroup symbol from structure data and store in attrs. Modifies in-place."""
    for entry in entries:
        s = _optimade_to_structure(entry["attrs"])
        if s is None:
            continue
        try:
            entry["attrs"]["_computed_spacegroup"] = SpacegroupAnalyzer(s).get_space_group_symbol()
        except Exception:
            pass


def _query_jarvis_sync(formula_reduced: str) -> tuple[list[dict], str | None]:
    if not jarvis_cache.is_available():
        return [], "JARVIS cache not loaded"
    raw = jarvis_cache.query(formula_reduced, max_results=_MAX_CANDIDATES)
    entries = []
    for item in raw:
        jid = item.get("jid", "")
        formula = item.get("formula", "")
        spg = _spg_number_to_symbol(item.get("spg"))
        url = f"https://www.ctcms.nist.gov/~knc6/static/JARVIS-DFT/{jid}.xml" if jid else None
        if url:
            entries.append({"item": item, "attrs": {"spg": spg, "jid": jid, "formula": formula}, "url": url})
    return entries, None


# ── Conversion to MatchResult (without StructureMatcher) ─────────────────────

def _optimade_entry_to_result(entry: dict, source: str) -> MatchEntry:
    attrs = entry["attrs"]
    item = entry["item"]
    formula = attrs.get("chemical_formula_reduced", "")

    if source == "Materials Project":
        spg = attrs.get("_computed_spacegroup")
        entry_id = item.get("id")
    elif source == "Materials Cloud":
        spg = attrs.get("_computed_spacegroup")
        entry_id = attrs.get("_mcloud_mc3d_id")
    else:  # NOMAD
        spg = attrs.get("_nmd_results_material_symmetry_space_group_symbol") or attrs.get("_computed_spacegroup")
        entry_id = item.get("id")

    return MatchEntry(
        formula=formula,
        spacegroup=spg,
        entry_id=entry_id,
        source=source,
        url=entry["url"],
        matched=False,
        score=None,
    )


def _jarvis_entry_to_result(entry: dict) -> MatchEntry:
    attrs = entry["attrs"]
    return MatchEntry(
        formula=attrs.get("formula", ""),
        spacegroup=attrs.get("spg"),
        entry_id=attrs.get("jid"),
        source="JARVIS",
        url=entry["url"],
        matched=False,
        score=None,
    )


# ── File-mode: match by formula + spacegroup ─────────────────────────────────

def _match_by_formula_spg(
    query_formula: str,
    query_spg: str | None,
    optimade_entries: list[tuple[str, list[dict]]],  # (source, entries)
    jarvis_entries: list[dict],
) -> list[MatchEntry]:
    results: list[MatchEntry] = []

    for source, entries in optimade_entries:
        for entry in entries:
            attrs = entry["attrs"]
            formula = attrs.get("chemical_formula_reduced", "")
            if source == "Materials Project":
                spg = attrs.get("_computed_spacegroup")
                entry_id = entry["item"].get("id")
            elif source == "Materials Cloud":
                spg = attrs.get("_computed_spacegroup")
                entry_id = attrs.get("_mcloud_mc3d_id")
            else:  # NOMAD
                spg = attrs.get("_nmd_results_material_symmetry_space_group_symbol") or attrs.get("_computed_spacegroup")
                entry_id = entry["item"].get("id")
            matched = (
                formula == query_formula
                and spg is not None
                and query_spg is not None
                and spg == query_spg
            )
            results.append(MatchEntry(
                formula=formula,
                spacegroup=spg,
                entry_id=entry_id,
                source=source,
                url=entry["url"],
                matched=matched,
                score=1.0 if matched else None,
            ))

    for entry in jarvis_entries:
        attrs = entry["attrs"]
        formula = attrs.get("formula", "")
        spg = attrs.get("spg")
        matched = (
            formula == query_formula
            and spg is not None
            and query_spg is not None
            and spg == query_spg
        )
        results.append(MatchEntry(
            formula=formula,
            spacegroup=spg,
            entry_id=attrs.get("jid"),
            source="JARVIS",
            url=entry["url"],
            matched=matched,
            score=1.0 if matched else None,
        ))

    results.sort(key=lambda r: (not r.matched, r.source, r.formula))
    return results


# ── Main route ────────────────────────────────────────────────────────────────

@router.post("/structure-match", response_model=StructureMatchResponse)
async def structure_match(req: StructureMatchRequest):
    errors: dict[str, str] = {}

    if req.mode == "file":
        if not req.structure_content or not req.structure_name:
            raise HTTPException(400, "structure_content and structure_name required for file mode")
        query_structure = _parse_structure(req.structure_content, req.structure_name)
        features = extract_match_features(query_structure)
        elements = features.elements
        query_formula = _to_optimade_formula(features.formula_reduced)
        try:
            query_spg = SpacegroupAnalyzer(query_structure).get_space_group_symbol()
        except Exception:
            query_spg = None
        logger.info("file mode: formula=%s spg=%s", query_formula, query_spg)
    elif req.mode == "formula":
        if not req.formula:
            raise HTTPException(400, "formula required for formula mode")
        raw = req.formula.strip()
        comp = None
        for attempt in [raw, raw.title()]:
            try:
                comp = Composition(attempt)
                break
            except Exception:
                continue
        if comp is None:
            raise HTTPException(400, f"Cannot parse formula: {raw!r}")
        elements = sorted(comp.chemical_system.split("-"))
        query_formula = comp.reduced_formula
        query_structure = None
    else:
        raise HTTPException(400, f"Unknown mode: {req.mode!r}")

    # Parallel DB queries
    # MP uses mp-api (has nsites + spacegroup natively), no structure fetch needed.
    # MC has no spacegroup field → structure always fetched for local SpacegroupAnalyzer.
    # NOMAD has native spacegroup field → no structure fetch needed.
    async with httpx.AsyncClient(timeout=_OPTIMADE_TIMEOUT) as client:
        mp_task = _query_mp(query_formula, client)
        mc_task = _query_mc(query_formula, client)
        nomad_task = _query_nomad(query_formula, client)

        (mp_entries, mp_err), (mc_entries, mc_err), (nomad_entries, nomad_err) = (
            await asyncio.gather(mp_task, mc_task, nomad_task)
        )

    if mp_err:
        errors["Materials Project"] = mp_err
    if mc_err:
        errors["Materials Cloud"] = mc_err
    if nomad_err:
        errors["NOMAD"] = nomad_err

    jarvis_entries, jarvis_err = _query_jarvis_sync(query_formula)
    if jarvis_err:
        errors["JARVIS"] = jarvis_err

    # Compute spacegroup locally for MC (no spacegroup field in their OPTIMADE)
    if mc_entries:
        _enrich_with_spacegroup(mc_entries)

    # Build results
    if req.mode == "file" and query_structure is not None:
        results = _match_by_formula_spg(
            query_formula,
            query_spg,
            [
                ("Materials Project", mp_entries),
                ("Materials Cloud", mc_entries),
                ("NOMAD", nomad_entries),
            ],
            jarvis_entries,
        )
    else:
        results = (
            [_optimade_entry_to_result(e, "Materials Project") for e in mp_entries]
            + [_optimade_entry_to_result(e, "Materials Cloud") for e in mc_entries]
            + [_optimade_entry_to_result(e, "NOMAD") for e in nomad_entries]
            + [_jarvis_entry_to_result(e) for e in jarvis_entries]
        )

    return StructureMatchResponse(
        results=results,
        query_formula=query_formula,
        errors=errors,
    )
