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
import os
from typing import Any, Optional, Type, TypeVar, Union

from dotenv import load_dotenv
from litellm import acompletion, completion
from pydantic import BaseModel, ValidationError

# LangSmith tracing — these decorators no-op gracefully if the API key is
# missing, so the agent stays runnable even without observability configured.
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

load_dotenv()

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "False").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Single model for all nodes in v1 (decision 1.2). Production note: cheaper
# nodes (plan / critique) could be routed to a smaller/local model.
DEFAULT_REMOTE_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_LOCAL_MODEL = "ollama/mistral:latest"
LOCAL_API_BASE = "http://localhost:11434"

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
        kwargs["api_key"] = GEMINI_API_KEY
    return kwargs


def _content(response: Any) -> str:
    return response["choices"][0]["message"]["content"]


def _usage(response: Any) -> dict:
    """Extract token usage from a litellm response in a forgiving way."""
    try:
        u = response.get("usage") if isinstance(response, dict) else response.usage
    except Exception:
        return {}
    if u is None:
        return {}
    if hasattr(u, "model_dump"):
        return u.model_dump()
    if hasattr(u, "__dict__"):
        return {k: v for k, v in u.__dict__.items() if not k.startswith("_")}
    return dict(u)


def _attach_metadata(*, model: str, usage: dict, tags: Optional[list[str]]) -> None:
    """Attach model/usage/tag info to the current LangSmith run tree (if any).

    Decision 7.3: tokens become queryable in LangSmith via run metadata.
    """
    rt = get_current_run_tree()
    if rt is None:
        return
    extra = {"model": model}
    if usage:
        extra["usage"] = usage
    rt.metadata = (rt.metadata or {}) | extra
    if tags:
        rt.tags = list(set((rt.tags or []) + tags))


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
def chat(
    messages: list[dict],
    *,
    response_model: Optional[Type[T]] = None,
    temperature: float = 0.4,
    tags: Optional[list[str]] = None,
) -> Union[str, T]:
    """Sync LLM call. See module docstring."""
    if response_model is None:
        kwargs = _build_kwargs(messages, temperature, tags)
        response = completion(**kwargs)
        _attach_metadata(model=kwargs["model"], usage=_usage(response), tags=tags)
        return _content(response)

    schema = response_model.model_json_schema()
    augmented = _augment_for_json_mode(messages, schema)

    last_error: Optional[Exception] = None
    for _ in range(2):  # initial + 1 retry
        kwargs = _build_kwargs(augmented, temperature, tags)
        response = completion(**kwargs)
        _attach_metadata(model=kwargs["model"], usage=_usage(response), tags=tags)
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


# ---------------------------------------------------------------------------
# Async entry point — used by the LangGraph nodes
# ---------------------------------------------------------------------------


@traceable(run_type="llm", name="litellm.achat")
async def achat(
    messages: list[dict],
    *,
    response_model: Optional[Type[T]] = None,
    temperature: float = 0.4,
    tags: Optional[list[str]] = None,
) -> Union[str, T]:
    """Async LLM call. Mirrors ``chat`` exactly but uses ``litellm.acompletion``.

    LangGraph nodes are async (decision 5.1) so they ``await achat(...)``
    instead of calling ``chat(...)``. Behavior, retry policy, and tracing
    are identical to the sync path.
    """
    if response_model is None:
        kwargs = _build_kwargs(messages, temperature, tags)
        response = await acompletion(**kwargs)
        _attach_metadata(model=kwargs["model"], usage=_usage(response), tags=tags)
        return _content(response)

    schema = response_model.model_json_schema()
    augmented = _augment_for_json_mode(messages, schema)

    last_error: Optional[Exception] = None
    for _ in range(2):
        kwargs = _build_kwargs(augmented, temperature, tags)
        response = await acompletion(**kwargs)
        _attach_metadata(model=kwargs["model"], usage=_usage(response), tags=tags)
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


# ---------------------------------------------------------------------------
# Smoke tests — `python llm_client.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    print(f"LOCAL_MODEL={LOCAL_MODEL}")

    # 1) Free-form sync
    text = chat(
        [{"role": "user", "content": "Reply with the single word: ready"}],
        temperature=0.0,
    )
    print("sync free-form:", text.strip())

    # 2) Structured async
    class Ping(BaseModel):
        message: str
        ok: bool

    async def run_async():
        return await achat(
            [{"role": "user", "content": "Say ok with message 'hello'."}],
            response_model=Ping,
            temperature=0.0,
        )

    parsed = asyncio.run(run_async())
    print("async structured:", parsed)
