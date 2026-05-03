"""LLM client.

A thin wrapper around `litellm.completion` that supports two modes:

1. Free-form text response (`response_model=None`).
2. Structured response — pass a Pydantic `BaseModel` subclass and we will
   prompt the model to return JSON, validate it with `model_validate_json`,
   and retry once on failure.

We use the *manual JSON-mode + Pydantic + 1 retry* approach so the behavior
is transparent and easy to explain. Instructor / outlines / native
`response_format` are reasonable production upgrades.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Type, TypeVar, Union

from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel, ValidationError

load_dotenv()

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "False").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Single model for all nodes in v1 (decision 1.2).
# Production note: cheaper nodes (plan / critique) could be routed to a
# smaller/local model.
DEFAULT_REMOTE_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_LOCAL_MODEL = "ollama/mistral:latest"
LOCAL_API_BASE = "http://localhost:11434"

T = TypeVar("T", bound=BaseModel)


def _build_kwargs(messages: list[dict], temperature: float, tags: Optional[list[str]]) -> dict:
    """Pick model + auth based on LOCAL_MODEL env."""
    kwargs: dict = {
        "messages": messages,
        "temperature": temperature,
    }
    if tags:
        # litellm forwards `metadata` to LangSmith when tracing is on.
        kwargs["metadata"] = {"tags": tags}

    if LOCAL_MODEL:
        kwargs["model"] = DEFAULT_LOCAL_MODEL
        kwargs["api_base"] = LOCAL_API_BASE
    else:
        kwargs["model"] = DEFAULT_REMOTE_MODEL
        kwargs["api_key"] = GEMINI_API_KEY
    return kwargs


def _content(response) -> str:
    return response["choices"][0]["message"]["content"]


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
        # remove first fence line
        text = text.split("\n", 1)[1] if "\n" in text else text
        # remove trailing fence
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


def chat(
    messages: list[dict],
    *,
    response_model: Optional[Type[T]] = None,
    temperature: float = 0.4,
    tags: Optional[list[str]] = None,
) -> Union[str, T]:
    """Call the LLM.

    If `response_model` is provided, the LLM is instructed to return JSON
    matching that schema; we validate with one retry on failure.
    """
    if response_model is None:
        response = completion(**_build_kwargs(messages, temperature, tags))
        return _content(response)

    schema = response_model.model_json_schema()
    instruction = _json_instruction(schema)

    # Append the JSON instruction to the first system message, or insert one.
    augmented = list(messages)
    if augmented and augmented[0].get("role") == "system":
        augmented[0] = {
            **augmented[0],
            "content": augmented[0]["content"] + "\n\n" + instruction,
        }
    else:
        augmented.insert(0, {"role": "system", "content": instruction})

    last_error: Optional[Exception] = None
    for attempt in range(2):  # initial + 1 retry
        response = completion(**_build_kwargs(augmented, temperature, tags))
        raw = _strip_fences(_content(response))
        try:
            return response_model.model_validate_json(raw)
        except ValidationError as e:
            last_error = e
            # On retry, show the model what went wrong so it can self-correct.
            augmented = augmented + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your previous response failed JSON schema validation "
                        f"with these errors:\n{e}\n"
                        "Return a corrected JSON object now. JSON only."
                    ),
                },
            ]

    raise RuntimeError(
        f"LLM did not produce valid {response_model.__name__} JSON after retry. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Smoke tests — `python llm_client.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"LOCAL_MODEL={LOCAL_MODEL}")

    # 1) Free-form
    text = chat(
        [{"role": "user", "content": "Reply with the single word: ready"}],
        temperature=0.0,
    )
    print("free-form:", text.strip())

    # 2) Structured
    class Ping(BaseModel):
        message: str
        ok: bool

    parsed = chat(
        [{"role": "user", "content": "Say ok with message 'hello'."}],
        response_model=Ping,
        temperature=0.0,
    )
    print("structured:", parsed)
