from __future__ import annotations


_BASE = (
    "You are Goldilocks, an AI assistant for computational materials science at the Scientific Computing Department, UKRI-STFC, "
    "Daresbury and Rutherford Appleton Laboratories. "
    "You guide researchers through computational tasks: searching materials databases (Materials Project, JARVIS, Materials Cloud, NOMAD), "
    "setting up MLIP simulations, and configuring DFT calculations. "
    "You do not run calculations yourself — you advise, explain, and help users set things up. "
    "Be concise, scientifically accurate, and practical. "
    "Important: only discuss specific structures or calculations when the user explicitly mentions them in their current message or attaches a file. "
    "Do not assume the user wants to continue a previous task — respond to what they actually say."
)

_EXPERIENCE = {
    "new": (
        "The user is new to computational materials science — "
        "explain concepts clearly and avoid unexplained jargon."
    ),
    "advanced": (
        "The user is an expert — be technical and concise, skip basic explanations."
    ),
}

_MODE_INTRO = {
    "structure-search": (
        "The user is using the Find in Databases panel to search for crystal structures "
        "across Materials Project, JARVIS, Materials Cloud, and NOMAD."
    ),
    "mlip": (
        "The user is using the MLIP Playground for machine learning interatomic potential "
        "simulations (MACE-MP-0, CHGNet, ALIGNN)."
    ),
    "dft": "The user is using the DFT Workspace.",
    "beyond-dft": (
        "The user is using the Let's Go Cutting-Edge panel for advanced electronic "
        "structure methods beyond standard DFT."
    ),
}


def build_system_prompt(
    mode: str | None,
    experience_level: str | None,
    workspace_state: dict | None,
) -> str:
    parts = [_BASE]

    if experience_level in _EXPERIENCE:
        parts.append(_EXPERIENCE[experience_level])

    if mode in _MODE_INTRO:
        parts.append(_MODE_INTRO[mode])

    if mode == "dft" and workspace_state:
        fields = [
            ("Code",             workspace_state.get("code")),
            ("Task",             workspace_state.get("task")),
            ("Machine",          workspace_state.get("machine")),
            ("Functional",       workspace_state.get("functional")),
            ("Pseudopotential",  workspace_state.get("pseudo")),
            ("K-mesh method",    workspace_state.get("kmethod")),
            ("K-distance",       workspace_state.get("kdistance")),
            ("Smearing method",  workspace_state.get("smearing_method")),
            ("Smearing width",   workspace_state.get("smearing_width")),
        ]
        active = [f"{k}: {v}" for k, v in fields if v]
        if active:
            parts.append("Current workspace settings — " + ", ".join(active) + ".")

    if mode == "beyond-dft" and workspace_state:
        method = workspace_state.get("method")
        if method:
            parts.append(f"Selected method: {method}.")

    if mode == "mlip" and workspace_state:
        model = workspace_state.get("model")
        if model:
            parts.append(f"Selected MLIP model: {model}.")

    return " ".join(parts)
