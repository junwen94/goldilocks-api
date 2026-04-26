"""JARVIS DFT-3D local cache — download once, query in-process."""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_FIGSHARE_URL = "https://ndownloader.figshare.com/files/38521619"
_CACHE_DIR = Path.home() / ".cache" / "goldilocks"
_CACHE_FILE = _CACHE_DIR / "jarvis_dft_3d.json"

_entries: list[dict] | None = None


def is_available() -> bool:
    return _entries is not None


def load() -> None:
    """Load JARVIS data from disk cache into memory. No-op if already loaded."""
    global _entries
    if _entries is not None:
        return
    if not _CACHE_FILE.exists():
        logger.warning("JARVIS cache not found at %s — run download_cache() first", _CACHE_FILE)
        return
    logger.info("Loading JARVIS DFT-3D cache from %s …", _CACHE_FILE)
    with open(_CACHE_FILE) as f:
        _entries = json.load(f)
    logger.info("Loaded %d JARVIS entries", len(_entries))


async def download_cache() -> None:
    """Download the JARVIS DFT-3D dataset from Figshare and save to disk cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading JARVIS DFT-3D from %s …", _FIGSHARE_URL)
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(_FIGSHARE_URL, follow_redirects=True)
        response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        json_name = next(n for n in zf.namelist() if n.endswith(".json"))
        data = json.loads(zf.read(json_name))

    with open(_CACHE_FILE, "w") as f:
        json.dump(data, f)

    global _entries
    _entries = data
    logger.info("JARVIS cache saved (%d entries)", len(_entries))


def get_by_jid(jid: str) -> dict | None:
    """Return the full JARVIS entry for a given JID, or None if not found."""
    if _entries is None:
        return None
    for entry in _entries:
        if entry.get("jid") == jid:
            return entry
    return None


def query(formula_reduced: str, max_results: int = 20) -> list[dict]:
    """Return JARVIS entries whose reduced formula matches the query."""
    from pymatgen.core import Composition

    if _entries is None:
        return []

    try:
        target = Composition(formula_reduced).reduced_formula
    except Exception:
        return []

    results = []
    for entry in _entries:
        try:
            entry_formula = entry.get("formula", "")
            entry_reduced = Composition(entry_formula).reduced_formula
        except Exception:
            continue
        if entry_reduced == target:
            results.append(entry)
            if len(results) >= max_results:
                break
    return results
