"""
Adapted copy of Backend/llm_client.py.

Same litellm-based logic (Gemini in cloud mode, Ollama/mistral in local mode),
but exposed as reusable functions so LangGraph nodes can call them.

- `chat(...)`            — plain text response.
- `chat_structured(...)` — JSON-mode + Pydantic validation + 1 repair retry.

Both are wrapped in @traceable so every call is a span in the LangSmith trace.
"""

import json
import os
import re
from typing import Type, TypeVar

from dotenv import load_dotenv
from langsmith import traceable
from litellm import completion
from pydantic import BaseModel, ValidationError

load_dotenv()
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "False").lower() == "true"

T = TypeVar("T", bound=BaseModel)


def _completion(messages, *, temperature, response_format=None):
    """Single dispatch point so chat() and chat_structured() share routing."""
    if LOCAL_MODEL:
        return completion(
            model="ollama/mistral:latest",
            messages=messages,
            api_base="http://localhost:11434",
            temperature=temperature,
            response_format=response_format,
        )
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Put it in a .env file or export it, "
            "or set LOCAL_MODEL=true to use Ollama."
        )
    return completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=messages,
        api_key=gemini_api_key,
        temperature=temperature,
        response_format=response_format,
    )


@traceable(run_type="llm", name="llm_client.chat")
def chat(prompt: str, *, system: str | None = None, temperature: float = 0.7) -> str:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if LOCAL_MODEL:
        response = completion(
            model="ollama/mistral:latest",
            messages=messages,
            api_base="http://localhost:11434",
            temperature=temperature,
        )
    else:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in a .env file or export it, "
                "or set LOCAL_MODEL=true to use Ollama."
            )
        response = completion(
            # --- Active Model ---
            # model="gemini/gemini-2.5-flash",
            
            # --- Fallback Gemini Models (Uncomment to switch) ---
            # model="gemini/gemini-3.0-flash",           # Newest ultra-fast model
            model="gemini/gemini-2.5-flash-lite",      # Most lightweight and cost-effective
            # model="gemini/gemini-3.1-flash-lite",      # Advanced lightweight model
            # model="gemini/gemini-2.5-pro",             # High-reasoning model (free tier limits apply)
            # model="gemini/gemini-3.1-pro-preview",     # Preview of advanced reasoning model
            # model="gemini/gemini-2.0-flash",           # Older reliable fallback
            
            messages=messages,
            api_key=gemini_api_key,
            temperature=temperature,
        )

    return response["choices"][0]["message"]["content"]


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Some models wrap JSON in ```json ... ``` even when told not to."""
    return _FENCE_RE.sub("", text).strip()


@traceable(run_type="llm", name="llm_client.chat_structured")
def chat_structured(
    prompt: str,
    response_model: Type[T],
    *,
    system: str | None = None,
    temperature: float = 0.2,
    max_retries: int = 1,
) -> T:
    """
    Ask the LLM for JSON that conforms to `response_model`, validate with
    Pydantic, retry once with the validation error fed back to the model
    if it fails.

    Production knobs that matter (and are wired in):
      - `temperature` defaults to 0.2 because shape stability beats creativity.
      - JSON-mode (`response_format={"type": "json_object"}`) so providers
        return parseable JSON.
      - Schema is generated from the Pydantic model and injected into the
        system prompt. Field descriptions become model-visible hints.
      - Markdown fences are stripped defensively.
      - On ValidationError we retry, including the error text so the model
        knows what to fix. This recovers the vast majority of failures.
    """
    schema = response_model.model_json_schema()
    schema_block = json.dumps(schema, indent=2)

    schema_instructions = (
        "You MUST respond with a single JSON object that conforms exactly to "
        "this JSON Schema:\n\n"
        f"{schema_block}\n\n"
        "Rules:\n"
        "- Output JSON only. No prose before or after.\n"
        "- No markdown code fences.\n"
        "- Every required field must be present.\n"
        "- Do not invent fields that are not in the schema."
    )
    full_system = f"{system}\n\n{schema_instructions}" if system else schema_instructions

    messages: list[dict] = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": prompt},
    ]

    last_error: str | None = None
    last_raw: str | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response failed validation:\n"
                        f"{last_error}\n\n"
                        f"Your previous response was:\n{last_raw}\n\n"
                        "Return a corrected JSON object only."
                    ),
                }
            )

        response = _completion(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        choice = response["choices"][0]
        finish_reason = choice.get("finish_reason")
        last_raw = choice["message"]["content"]

        if finish_reason == "length":
            raise RuntimeError(
                "LLM response was truncated (finish_reason=length). "
                "Increase max_tokens or shorten the schema."
            )

        try:
            cleaned = _strip_fences(last_raw)
            return response_model.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError) as e:
            last_error = str(e)
            if attempt == max_retries:
                raise RuntimeError(
                    f"Structured output failed after {max_retries + 1} attempts.\n"
                    f"Last error: {last_error}\nLast raw output: {last_raw}"
                ) from e

    raise RuntimeError("unreachable")


if __name__ == "__main__":
    print("hello from llm_client.py")
    print("LOCAL_MODEL =", LOCAL_MODEL)
    print("Response:", chat("respond in 20 words. who are you?"))

    from schemas import WordAnalysis

    result = chat_structured(
        "Analyze the word: 'serendipity'.",
        response_model=WordAnalysis,
    )
    print("\nStructured:")
    print(result.model_dump_json(indent=2))
