"""Tool definitions and execution for the Goldilocks agent tool loop."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# ── Tool schemas (OpenAI function-calling format) ─────────────────────────────

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_materials_databases",
            "description": (
                "Search for crystal structures by chemical formula across Materials Project, "
                "JARVIS, Materials Cloud, and NOMAD. "
                "Returns matching entries with formula, spacegroup, and a direct database URL. "
                "Use this whenever the user asks to find, look up, or search for a material."
            ),
            "parameters": {
                "type": "object",
                "required": ["formula"],
                "properties": {
                    "formula": {
                        "type": "string",
                        "description": "Chemical formula, e.g. 'Fe2O3' or 'TiO2'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_kpoints",
            "description": (
                "Compute a k-point mesh for a crystal structure given a desired k-spacing density. "
                "Returns the mesh dimensions (Nx×Ny×Nz) and the number of irreducible k-points. "
                "Use this when the user asks about k-mesh, k-points, or sampling density for a structure."
            ),
            "parameters": {
                "type": "object",
                "required": ["structure_content", "structure_name", "kdistance_id"],
                "properties": {
                    "structure_content": {
                        "type": "string",
                        "description": "Full text content of the structure file.",
                    },
                    "structure_name": {
                        "type": "string",
                        "description": "Filename of the structure, e.g. 'Fe2O3.cif'.",
                    },
                    "kdistance_id": {
                        "type": "string",
                        "enum": ["gamma", "light", "standard", "dense", "very-dense"],
                        "description": (
                            "'gamma'=Γ-only; 'light'≈0.40 Å⁻¹; 'standard'≈0.25 Å⁻¹; "
                            "'dense'≈0.15 Å⁻¹; 'very-dense'≈0.08 Å⁻¹."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_pseudopotentials",
            "description": (
                "Find pseudopotential files for a crystal structure and DFT functional from the "
                "local pseudopotential library. Returns per-element candidate files. "
                "Use this when the user asks which pseudopotentials to use for a DFT calculation."
            ),
            "parameters": {
                "type": "object",
                "required": ["structure_content", "structure_name", "functional"],
                "properties": {
                    "structure_content": {
                        "type": "string",
                        "description": "Full text content of the structure file.",
                    },
                    "structure_name": {
                        "type": "string",
                        "description": "Filename of the structure, e.g. 'Fe2O3.cif'.",
                    },
                    "functional": {
                        "type": "string",
                        "description": "DFT functional, e.g. 'PBE', 'PBEsol', 'LDA'.",
                    },
                    "pseudo_type": {
                        "type": "string",
                        "description": "Optional pseudopotential type: 'NC', 'US', or 'PAW'.",
                    },
                },
            },
        },
    },
]


# ── Tool execution ─────────────────────────────────────────────────────────────

async def execute_tool(name: str, arguments: dict) -> str:
    """Execute a named tool and return its result as a JSON string."""
    logger.info("Tool call: %s(%s)", name, list(arguments.keys()))
    try:
        if name == "search_materials_databases":
            return await _search_materials(arguments)
        if name == "suggest_kpoints":
            return await _suggest_kpoints(arguments)
        if name == "suggest_pseudopotentials":
            return await _suggest_pseudopotentials(arguments)
        return json.dumps({"error": f"Unknown tool: {name!r}"})
    except Exception as exc:
        logger.warning("Tool %r raised: %s", name, exc)
        return json.dumps({"error": str(exc)})


async def _search_materials(args: dict) -> str:
    from app.routes.structure_match import structure_match, StructureMatchRequest

    resp = await structure_match(StructureMatchRequest(mode="formula", formula=args["formula"]))

    if not resp.results:
        return json.dumps({"message": f"No results found for '{args['formula']}'.", "results": []})

    entries = []
    for r in resp.results[:12]:
        entry: dict = {"source": r.source, "formula": r.formula, "url": r.url}
        if r.spacegroup:
            entry["spacegroup"] = r.spacegroup
        if r.entry_id:
            entry["entry_id"] = r.entry_id
        entries.append(entry)

    return json.dumps(
        {
            "query_formula": resp.query_formula,
            "total_results": len(resp.results),
            "results": entries,
            **({"errors": resp.errors} if resp.errors else {}),
        }
    )


async def _suggest_kpoints(args: dict) -> str:
    from app.routes.dft import dft_kpoints, KpointsRequest

    resp = await dft_kpoints(
        KpointsRequest(
            structure_content=args["structure_content"],
            structure_name=args["structure_name"],
            kdistance_id=args["kdistance_id"],
        )
    )
    return json.dumps(
        {
            "grid": list(resp.grid),
            "n_reduced_kpoints": resp.n_reduced_kpoints,
            "k_distance": resp.k_distance,
            "summary": resp.summary,
        }
    )


async def _suggest_pseudopotentials(args: dict) -> str:
    from app.routes.dft import dft_pseudo, PseudoRequest

    resp = await dft_pseudo(
        PseudoRequest(
            structure_content=args["structure_content"],
            structure_name=args["structure_name"],
            functional=args["functional"],
            pseudo_type=args.get("pseudo_type"),
        )
    )
    candidates = {
        element: [p.model_dump() for p in pseudos]
        for element, pseudos in resp.candidates.items()
    }
    return json.dumps({"candidates": candidates, "summary": resp.summary})
