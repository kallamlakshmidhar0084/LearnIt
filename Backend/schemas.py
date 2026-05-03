"""Schemas for the poster agent.

Two kinds of schemas live here:

1. **Pydantic `BaseModel`s** — used at every LLM boundary so structured
   outputs are validated. These are the *contracts* between nodes and the
   LLM; they must stay stable.

2. **`TypedDict` graph state** — what flows between LangGraph nodes.
   LangGraph idiom is `TypedDict` (cheap, mutable, no validation). Each
   field is `total=False`-style optional via `NotRequired` because
   different nodes populate different fields.
"""

from __future__ import annotations

from typing import Literal, NotRequired, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. LLM-facing Pydantic schemas
# ---------------------------------------------------------------------------


class PosterBrief(BaseModel):
    """Parsed user brief. Produced by `parse_brief` node."""

    raw_prompt: str = Field(..., description="Original prompt as received.")
    title: str = Field(..., description="Poster title / headline.")
    audience: Optional[str] = Field(None, description="Target audience, if given.")
    style: Optional[str] = Field(None, description="Style hint, e.g. 'bold retro'.")
    details: Optional[str] = Field(None, description="Free-form extra details.")
    user_specified_palette: bool = Field(
        False,
        description=(
            "True if the user explicitly mentioned colors / a palette in their "
            "brief. When True, the design node should NOT call pick_palette."
        ),
    )


class DesignSpec(BaseModel):
    """Design plan produced by `design` node before HTML/CSS is written."""

    mood: str = Field(..., description="One-word mood, e.g. 'bold', 'minimal'.")
    palette: list[str] = Field(
        ..., min_length=3, max_length=8,
        description="Hex colors, ordered [bg, surface, accent, text, muted, ...].",
    )
    font_pair: dict[str, str] = Field(
        ..., description="Mapping with 'heading' and 'body' font names."
    )
    layout: Literal["hero", "grid", "split", "minimal"] = Field(
        ..., description="High-level layout family."
    )
    rationale: str = Field(
        ..., description="1-2 sentence reasoning for these choices."
    )


class PosterDraft(BaseModel):
    """The HTML+CSS pair returned to the frontend."""

    html: str = Field(..., description="HTML body fragment for the poster.")
    css: str = Field(..., description="Standalone CSS string.")


class Critique(BaseModel):
    """Self-critique. Fixed checklist for deterministic eval scoring."""

    contrast_ok: bool
    alignment_ok: bool
    readability_ok: bool
    prompt_fidelity_ok: bool
    issues: list[str] = Field(
        default_factory=list, description="Concrete problems to fix."
    )

    @property
    def must_fix(self) -> bool:
        return not (
            self.contrast_ok
            and self.alignment_ok
            and self.readability_ok
            and self.prompt_fidelity_ok
        )


# ---------------------------------------------------------------------------
# 2. LangGraph state (TypedDict)
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    """State that flows between graph nodes.

    Use `NotRequired` for fields that are populated by later nodes, so the
    initial state can be minimal.
    """

    # Inputs from the API layer
    raw_prompt: str
    session_id: str
    use_memory: bool
    history: NotRequired[list[str]]  # prior prompts when use_memory=True

    # Populated by nodes
    brief: NotRequired[PosterBrief]
    design: NotRequired[DesignSpec]
    draft: NotRequired[PosterDraft]
    critique: NotRequired[Critique]

    # Loop control
    revisions_left: int

    # Final output (set on terminal node)
    final: NotRequired[PosterDraft]


# ---------------------------------------------------------------------------
# Smoke tests — `python schemas.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    brief = PosterBrief(raw_prompt="x", title="Demo", user_specified_palette=False)
    design = DesignSpec(
        mood="bold",
        palette=["#000", "#111", "#fff"],
        font_pair={"heading": "Inter", "body": "Inter"},
        layout="hero",
        rationale="bold + readable",
    )
    draft = PosterDraft(html="<div/>", css="*{}")
    critique = Critique(
        contrast_ok=True,
        alignment_ok=True,
        readability_ok=False,
        prompt_fidelity_ok=True,
        issues=["body text too small"],
    )

    state: GraphState = {
        "raw_prompt": "demo",
        "session_id": "s1",
        "use_memory": False,
        "revisions_left": 1,
        "brief": brief,
        "design": design,
        "draft": draft,
        "critique": critique,
    }

    assert critique.must_fix is True
    print("schemas OK")
    print("brief:", brief.model_dump())
    print("design:", design.model_dump())
    print("critique.must_fix:", critique.must_fix)
    print("state keys:", sorted(state.keys()))
