"""
================================================================================
                          PosterAgent — async wrapper
================================================================================

This is the only thing FastAPI talks to. It hides:
  - the LangGraph compile (done once at import time in agent_graph.py)
  - the in-memory session store (decision 0.2)
  - LangSmith config metadata propagation (decision 7.4)
  - the API response shape (matches `GenerateResponse` in main.py)

The class exposes a single async method, ``arun``, so FastAPI handlers can
``await poster_agent.arun(...)`` directly. We chose async (decision 5.1) so
that adding streaming later — per-node status events, partial designs — is a
local change inside this class, not a refactor across the stack.
"""

from __future__ import annotations

import uuid
from typing import Optional

from agent_graph import GRAPH
from schemas import GraphState, PosterDraft


class PosterAgent:
    """Thin async wrapper around the compiled LangGraph."""

    def __init__(self) -> None:
        # In-memory session store. Maps session_id -> list of prompts so the
        # agent can later use prior turns as context. Resets on process
        # restart (decision 0.2).
        self._sessions: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def arun(
        self,
        prompt: str,
        *,
        session_id: Optional[str],
        use_memory: bool,
    ) -> dict:
        """Run one poster-generation request through the graph.

        Returns a dict with the keys ``main.py``'s ``GenerateResponse`` model
        expects: session_id, html, css, used_memory, history_length,
        rationale, palette.
        """
        sid, used_memory, history = self._resolve_session(
            session_id=session_id, use_memory=use_memory, prompt=prompt
        )

        initial: GraphState = {
            "raw_prompt": prompt,
            "session_id": sid,
            "use_memory": used_memory,
            "history": history,
            "revisions_left": 1,
        }

        # LangGraph propagates this config to every child run. With LangSmith
        # tracing on, every node and LLM span ends up tagged with the session
        # id — so the demo can filter traces by session in the UI
        # (decision 7.4).
        config = {
            "metadata": {
                "session_id": sid,
                "use_memory": used_memory,
                "history_length": len(history),
            },
            "tags": [f"session:{sid}", "poster-agent:v1"],
            "run_name": f"poster-agent[{sid[:8]}]",
        }

        result: GraphState = await GRAPH.ainvoke(initial, config=config)
        return self._format_response(
            result=result, sid=sid, used_memory=used_memory, history=history
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_session(
        self,
        *,
        session_id: Optional[str],
        use_memory: bool,
        prompt: str,
    ) -> tuple[str, bool, list[str]]:
        """Pick (or mint) a session id, append the new prompt, return history."""
        if use_memory and session_id and session_id in self._sessions:
            sid = session_id
            self._sessions[sid].append(prompt)
            return sid, True, list(self._sessions[sid])

        sid = str(uuid.uuid4())
        self._sessions[sid] = [prompt]
        return sid, False, [prompt]

    def _format_response(
        self,
        *,
        result: GraphState,
        sid: str,
        used_memory: bool,
        history: list[str],
    ) -> dict:
        """Translate graph state into the API response shape."""
        final: PosterDraft = result["final"]

        rationale: Optional[str] = None
        palette: Optional[list[str]] = None
        design = result.get("design")
        if design is not None:
            rationale = design.rationale
            palette = design.palette

        return {
            "session_id": sid,
            "html": final.html,
            "css": final.css,
            "used_memory": used_memory,
            "history_length": len(history),
            "rationale": rationale,
            "palette": palette,
        }


# Single instance reused across requests — graph and session store stay alive
# for the process lifetime.
poster_agent = PosterAgent()
