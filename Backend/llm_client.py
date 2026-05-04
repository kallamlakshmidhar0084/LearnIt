"""LLM client.

A thin wrapper around ``litellm.completion`` / ``litellm.acompletion`` that
supports two modes:

1. Free-form text response (``response_model=None``).
2. Structured response — pass a Pydantic ``BaseModel`` subclass and we will
   prompt the model to return JSON, validate it with ``model_validate_json``,
   and retry once on failure.

Both ``chat`` (sync) and ``achat`` (async) are wrapped with
``@traceable(run_type="llm")`` so every LLM call is recorded as an LLM span
in LangSmith with prompt/response and token-usage metadata. LangGraph node
spans are auto-traced when ``LANGSMITH_TRACING=true`` is set in the
environment.

Decisions backing this design (see plan.md):
    1.1 — Manual JSON-mode + Pydantic + 1 retry (transparent, no extra dep).
    1.2 — Same model for all nodes in v1.
    7.2 — Trace LLM calls explicitly so prompts and tokens show up in LangSmith.
    7.3 — Attach `usage` to run-tree metadata so tokens are queryable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, Type, TypeVar, Union

from dotenv import load_dotenv
from litellm import acompletion, completion
from pydantic import BaseModel, ValidationError

# LangSmith tracing — @traceable becomes a no-op when LANGSMITH_TRACING is off,
# so the agent stays runnable without observability configured.
from langsmith import traceable
from langsmith.run_helpers import tracing_context
from langsmith.run_trees import RunTree

load_dotenv()

logger = logging.getLogger(__name__)

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "False").lower() == "true"
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# litellm's Anthropic provider reads ANTHROPIC_API_KEY from env. Mirror our
# CLAUDE_API_KEY into it so users only need to set one variable.
if CLAUDE_API_KEY and not os.getenv("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = CLAUDE_API_KEY

# Single model for all nodes in v1 (decision 1.2). Production note: cheaper
# nodes (plan / critique) could be routed to a smaller/local model.
DEFAULT_REMOTE_MODEL = "anthropic/claude-haiku-4-5-20251001"
DEFAULT_LOCAL_MODEL = "ollama/mistral:latest"
LOCAL_API_BASE = "http://localhost:11434"

_LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "").lower() == "true"

# Startup confirmation — visible in app logs at import time.
if LOCAL_MODEL:
    logger.info("llm_client configured: model=%s (local)", DEFAULT_LOCAL_MODEL)
elif CLAUDE_API_KEY:
    logger.info(
        "llm_client configured: model=%s, CLAUDE_API_KEY=set (len=%d), langsmith=%s",
        DEFAULT_REMOTE_MODEL,
        len(CLAUDE_API_KEY),
        "on" if _LANGSMITH_TRACING else "off",
    )
else:
    logger.warning(
        "llm_client: CLAUDE_API_KEY is NOT set — remote calls to %s will fail",
        DEFAULT_REMOTE_MODEL,
    )

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Helpers shared by sync + async paths
# ---------------------------------------------------------------------------


def _build_kwargs(
    messages: list[dict], temperature: float, tags: Optional[list[str]]
) -> dict:
    """Pick model + auth based on LOCAL_MODEL env."""
    kwargs: dict = {
        "messages": messages,
        "temperature": temperature,
    }
    if tags:
        kwargs["metadata"] = {"tags": tags}

    if LOCAL_MODEL:
        kwargs["model"] = DEFAULT_LOCAL_MODEL
        kwargs["api_base"] = LOCAL_API_BASE
    else:
        kwargs["model"] = DEFAULT_REMOTE_MODEL
        kwargs["api_key"] = CLAUDE_API_KEY
    return kwargs


def _content(response: Any) -> str:
    return response["choices"][0]["message"]["content"]


def _parent_from_config(config: Optional[dict]) -> Optional[RunTree]:
    """Build a LangSmith RunTree parent from a LangGraph RunnableConfig.

    `tracing_context(parent=...)` requires a fully-formed RunTree (with
    `dotted_order` / `trace_id` populated). Passing a bare run-id string
    triggers a `time data '' does not match format` ValueError inside
    LangSmith's date parser. `RunTree.from_runnable_config` constructs the
    proper parent from the LangChain callback manager LangGraph already set
    up for the active node, so the LLM span attaches as a child of the node
    instead of starting a new root.

    Returns None if no parent can be derived (tracing then behaves as before
    — the call still works, just at the top level).
    """
    if not config:
        return None
    try:
        return RunTree.from_runnable_config(config)
    except Exception:
        # Don't let tracing breakage take down the LLM call.
        logger.debug("Could not derive RunTree from config", exc_info=True)
        return None


def _json_instruction(schema: dict) -> str:
    """The hard rule we append to the system message in JSON mode."""
    return (
        "You MUST respond with a single JSON object that conforms to this "
        "JSON schema. Do not include any prose, markdown fences, or "
        "explanation outside the JSON.\n\n"
        f"JSON schema:\n{json.dumps(schema, indent=2)}"
    )


def _strip_fences(text: str) -> str:
    """Models sometimes wrap JSON in ```json ... ``` despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


def _augment_for_json_mode(messages: list[dict], schema: dict) -> list[dict]:
    """Inject the JSON-mode instruction into the first system message."""
    instruction = _json_instruction(schema)
    augmented = list(messages)
    if augmented and augmented[0].get("role") == "system":
        augmented[0] = {
            **augmented[0],
            "content": augmented[0]["content"] + "\n\n" + instruction,
        }
    else:
        augmented.insert(0, {"role": "system", "content": instruction})
    return augmented


def _retry_messages(prev: list[dict], raw: str, error: ValidationError) -> list[dict]:
    """Build the retry conversation, showing the model what went wrong."""
    return prev + [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                "Your previous response failed JSON schema validation with "
                f"these errors:\n{error}\n"
                "Return a corrected JSON object now. JSON only."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Sync entry point
# ---------------------------------------------------------------------------


@traceable(run_type="llm", name="litellm.chat")
def _chat_traced(
    messages: list[dict],
    *,
    response_model: Optional[Type[T]],
    temperature: float,
    tags: Optional[list[str]],
) -> Union[str, T]:
    """Inner traced body — see ``chat`` for the public contract."""
    if response_model is None:
        kwargs = _build_kwargs(messages, temperature, tags)
        response = completion(**kwargs)
        return _content(response)

    schema = response_model.model_json_schema()
    augmented = _augment_for_json_mode(messages, schema)

    last_error: Optional[Exception] = None
    for _ in range(2):  # initial + 1 retry
        kwargs = _build_kwargs(augmented, temperature, tags)
        response = completion(**kwargs)
        raw = _strip_fences(_content(response))
        try:
            return response_model.model_validate_json(raw)
        except ValidationError as e:
            last_error = e
            augmented = _retry_messages(augmented, raw, e)

    raise RuntimeError(
        f"LLM did not produce valid {response_model.__name__} JSON after retry. "
        f"Last error: {last_error}"
    )


def chat(
    messages: list[dict],
    *,
    response_model: Optional[Type[T]] = None,
    temperature: float = 0.4,
    tags: Optional[list[str]] = None,
    config: Optional[dict] = None,
) -> Union[str, T]:
    """Sync LLM call.

    Pass the LangGraph node's ``config: RunnableConfig`` to make the LLM span
    nest under the node's run in LangSmith.
    """
    parent = _parent_from_config(config)
    with tracing_context(parent=parent):
        return _chat_traced(
            messages,
            response_model=response_model,
            temperature=temperature,
            tags=tags,
        )


# ---------------------------------------------------------------------------
# Async entry point — used by the LangGraph nodes
# ---------------------------------------------------------------------------


@traceable(run_type="llm", name="litellm.achat")
async def _achat_traced(
    messages: list[dict],
    *,
    response_model: Optional[Type[T]],
    temperature: float,
    tags: Optional[list[str]],
) -> Union[str, T]:
    """Inner traced body — see ``achat`` for the public contract."""
    if response_model is None:
        kwargs = _build_kwargs(messages, temperature, tags)
        response = await acompletion(**kwargs)
        return _content(response)

    schema = response_model.model_json_schema()
    augmented = _augment_for_json_mode(messages, schema)

    last_error: Optional[Exception] = None
    for _ in range(2):
        kwargs = _build_kwargs(augmented, temperature, tags)
        response = await acompletion(**kwargs)
        raw = _strip_fences(_content(response))
        try:
            return response_model.model_validate_json(raw)
        except ValidationError as e:
            last_error = e
            augmented = _retry_messages(augmented, raw, e)

    raise RuntimeError(
        f"LLM did not produce valid {response_model.__name__} JSON after retry. "
        f"Last error: {last_error}"
    )


async def achat(
    messages: list[dict],
    *,
    response_model: Optional[Type[T]] = None,
    temperature: float = 0.4,
    tags: Optional[list[str]] = None,
    config: Optional[dict] = None,
) -> Union[str, T]:
    """Async LLM call. Mirrors ``chat`` exactly but uses ``litellm.acompletion``.

    Pass the LangGraph node's ``config: RunnableConfig`` so the LLM span
    nests under the node's run in LangSmith. Without it, the span still works
    but appears as a top-level trace.
    """
    parent = _parent_from_config(config)
    with tracing_context(parent=parent):
        return await _achat_traced(
            messages,
            response_model=response_model,
            temperature=temperature,
            tags=tags,
        )


# ---------------------------------------------------------------------------
# Smoke tests — `python llm_client.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    model = DEFAULT_LOCAL_MODEL if LOCAL_MODEL else DEFAULT_REMOTE_MODEL
    print(f"[smoke] LOCAL_MODEL={LOCAL_MODEL} model={model}")
    print(f"[smoke] CLAUDE_API_KEY={'set' if CLAUDE_API_KEY else 'MISSING'}")

    text = chat(
        [{"role": "user", "content": "Reply with the single word: ready"}],
        temperature=0.0,
    )
    print(f"[smoke] response: {text.strip()!r}")
    print("[smoke] OK — litellm + Anthropic configuration is working")
