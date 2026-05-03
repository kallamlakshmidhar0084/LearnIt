"""Agent tools.

Two tools live here for v1:

1. ``pick_template`` — **mandatory**. Returns a poster template description
   (informational, advertisement, caution, event, minimal). The LLM must
   always pick one of these so the generate node has a clear blueprint to
   follow.

2. ``pick_palette`` — **optional**. Returns a 5-color hex palette for a
   given mood. The LLM should call it only when the user has *not* already
   specified colors in their brief — that demonstrates LLM-driven,
   conditional tool-usage.

Both tools are plain Python callables. We orchestrate them manually from
``design_node`` (Pydantic ``ToolPlan`` → call function → feed result back),
so we don't need ``langchain_core.tools.@tool`` or a ``ToolNode``. LangSmith
still records the calls because they execute inside an auto-traced
LangGraph node span.

Both tools are deterministic (no network, no randomness) so traces and
RAGAS evaluations stay reproducible.
"""

from __future__ import annotations

from typing import Callable


# ---------------------------------------------------------------------------
# pick_template
# ---------------------------------------------------------------------------

# Every entry teaches the generate-node what kind of poster to compose.
# The hints are written as instructions the LLM can follow verbatim.
TEMPLATES: dict[str, dict[str, str]] = {
    "informational": {
        "name": "Informational",
        "purpose": "Convey facts, schedules, or instructions clearly.",
        "layout_hint": (
            "Strong title at the top, a short subtitle, then 2–4 fact "
            "blocks arranged in a grid. Footer with a source/credit line."
        ),
        "vibe_hint": "trustworthy, clear, generous whitespace, easy to scan",
    },
    "advertisement": {
        "name": "Advertisement",
        "purpose": "Sell a product or hype an event with a strong hook.",
        "layout_hint": (
            "Hero headline taking ~40% of the canvas, a single tagline, "
            "a prominent call-to-action chip, and 2 supporting feature cells."
        ),
        "vibe_hint": "bold, energetic, high-contrast, confident",
    },
    "caution": {
        "name": "Caution / Warning",
        "purpose": "Warn or alert the viewer about a hazard or rule.",
        "layout_hint": (
            "Centered warning emblem area at the top, an unambiguous bold "
            "warning headline, and 2–3 short safety bullet points underneath."
        ),
        "vibe_hint": "high-contrast, urgent, no decoration, unambiguous",
    },
    "event": {
        "name": "Event",
        "purpose": "Announce an event with date, venue, and atmosphere.",
        "layout_hint": (
            "Event title up top, a date+venue band in the middle, a tagline "
            "or lineup section, and a small footer with ticket info."
        ),
        "vibe_hint": "atmospheric, inviting, slightly dramatic",
    },
    "minimal": {
        "name": "Minimal",
        "purpose": "Single concept, quote, or art-piece poster.",
        "layout_hint": (
            "One headline, one accent shape or color block, lots of "
            "negative space. No grid, no decoration."
        ),
        "vibe_hint": "calm, premium, restrained, gallery-like",
    },
}


def pick_template(kind: str) -> dict:
    """Pick a poster template blueprint.

    The LLM MUST call this tool exactly once. Choose the ``kind`` that best
    fits the user's brief from this set:

    - "informational"  → facts, schedules, instructions
    - "advertisement"  → selling or hyping something
    - "caution"        → warnings, hazards, rules
    - "event"          → announcing an event with date/venue
    - "minimal"        → quote, single concept, art piece

    Returns a dict with the template's name, purpose, layout_hint, and
    vibe_hint. The generate node will use these hints to compose HTML+CSS.
    Unknown ``kind`` falls back to "informational".
    """
    key = (kind or "").strip().lower()
    if key not in TEMPLATES:
        key = "informational"
    return {"kind": key, **TEMPLATES[key]}


# ---------------------------------------------------------------------------
# pick_palette
# ---------------------------------------------------------------------------

# Curated palettes keyed by mood. Five hex codes per palette ordered as
# [bg, surface, accent, text, muted].
_PALETTES: dict[str, list[str]] = {
    "bold":      ["#0b1020", "#1f2937", "#f97316", "#f9fafb", "#9ca3af"],
    "minimal":   ["#ffffff", "#f4f4f5", "#111827", "#27272a", "#a1a1aa"],
    "retro":     ["#fef3c7", "#f59e0b", "#b45309", "#1c1917", "#78716c"],
    "vibrant":   ["#0f172a", "#1e1b4b", "#a855f7", "#fde047", "#94a3b8"],
    "elegant":   ["#0c0a09", "#1c1917", "#d4af37", "#fafaf9", "#a8a29e"],
    "playful":   ["#fce7f3", "#f9a8d4", "#7c3aed", "#0f172a", "#64748b"],
    "tech":      ["#020617", "#0f172a", "#22d3ee", "#e2e8f0", "#64748b"],
    "nature":    ["#064e3b", "#065f46", "#84cc16", "#f0fdf4", "#a7f3d0"],
    "warning":   ["#1c0a00", "#3f1d00", "#f59e0b", "#fffbeb", "#fde68a"],
    "default":   ["#0b1020", "#1f2937", "#6366f1", "#f9fafb", "#9ca3af"],
}


def _normalize(mood: str) -> str:
    if not mood:
        return "default"
    m = mood.strip().lower()
    for key in _PALETTES:
        if key in m:
            return key
    return "default"


def pick_palette(mood: str) -> dict:
    """Pick a 5-color hex palette for a mood.

    The LLM should call this tool ONLY when the user did not specify colors
    in their brief. Pass a short mood word — e.g. "bold", "minimal",
    "retro", "vibrant", "elegant", "playful", "tech", "nature", "warning".
    Unknown moods fall back to a sensible default.

    Returns a dict with ``mood`` (the matched key) and ``palette`` (a list
    of 5 hex codes ordered as [bg, surface, accent, text, muted]).
    """
    matched = _normalize(mood)
    return {"mood": matched, "palette": _PALETTES[matched]}


# ---------------------------------------------------------------------------
# Registry — consumed by the graph
# ---------------------------------------------------------------------------

TOOLS: dict[str, Callable[..., dict]] = {
    "pick_template": pick_template,
    "pick_palette": pick_palette,
}


# ---------------------------------------------------------------------------
# Smoke tests — `python tools.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t = pick_template(kind="advertisement")
    assert t["kind"] == "advertisement" and "layout_hint" in t
    print("pick_template OK ->", t)

    t_unknown = pick_template(kind="bogus")
    assert t_unknown["kind"] == "informational"
    print("pick_template fallback OK")

    p = pick_palette(mood="retro")
    assert p["mood"] == "retro" and len(p["palette"]) == 5
    print("pick_palette OK ->", p)

    print("registered tools:", list(TOOLS))
