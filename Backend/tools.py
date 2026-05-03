"""Agent tools.

For v1 we expose **one** tool — `pick_palette` — as a LangGraph-compatible
`@tool`. The LLM decides whether to call it: when the user has already
specified a palette in their brief, the LLM is instructed to skip the tool;
otherwise it should invoke `pick_palette(mood=...)` and use the returned
hex codes.

Keeping the tool deterministic (no network, no randomness) makes traces
reproducible and lets us assert on tool output during evals.
"""

from __future__ import annotations

from langchain_core.tools import tool

# Curated palettes keyed by mood. Five hex codes per palette: bg, surface,
# accent, text, muted.
_PALETTES: dict[str, list[str]] = {
    "bold":      ["#0b1020", "#1f2937", "#f97316", "#f9fafb", "#9ca3af"],
    "minimal":   ["#ffffff", "#f4f4f5", "#111827", "#27272a", "#a1a1aa"],
    "retro":     ["#fef3c7", "#f59e0b", "#b45309", "#1c1917", "#78716c"],
    "vibrant":   ["#0f172a", "#1e1b4b", "#a855f7", "#fde047", "#94a3b8"],
    "elegant":   ["#0c0a09", "#1c1917", "#d4af37", "#fafaf9", "#a8a29e"],
    "playful":   ["#fce7f3", "#f9a8d4", "#7c3aed", "#0f172a", "#64748b"],
    "tech":      ["#020617", "#0f172a", "#22d3ee", "#e2e8f0", "#64748b"],
    "nature":    ["#064e3b", "#065f46", "#84cc16", "#f0fdf4", "#a7f3d0"],
    "default":   ["#0b1020", "#1f2937", "#6366f1", "#f9fafb", "#9ca3af"],
}


def _normalize(mood: str) -> str:
    if not mood:
        return "default"
    m = mood.strip().lower()
    # Pick the first known mood whose key appears as a substring.
    for key in _PALETTES:
        if key in m:
            return key
    return "default"


@tool
def pick_palette(mood: str) -> dict:
    """Return a 5-color hex palette that matches the requested mood.

    Call this tool ONLY when the user has not already specified colors in
    their brief. Pass a short mood word (e.g. "bold", "minimal", "retro",
    "vibrant", "elegant", "playful", "tech", "nature"). Unknown moods fall
    back to a sensible default palette.

    Returns a dict with:
      - mood: the matched mood key
      - palette: list of 5 hex strings ordered as [bg, surface, accent, text, muted]
    """
    matched = _normalize(mood)
    return {"mood": matched, "palette": _PALETTES[matched]}


# Map exposed to the graph so `agent_graph.py` can build a ToolNode and the
# nodes can also call tools directly when needed.
TOOLS = [pick_palette]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# Smoke tests — `python tools.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Direct .invoke is the public LangChain tool API.
    out = pick_palette.invoke({"mood": "retro"})
    assert out["mood"] == "retro"
    assert len(out["palette"]) == 5
    assert all(c.startswith("#") and len(c) == 7 for c in out["palette"])

    out_unknown = pick_palette.invoke({"mood": "weirdmood"})
    assert out_unknown["mood"] == "default"

    print("pick_palette OK ->", out)
    print("registered tools:", list(TOOLS_BY_NAME))
