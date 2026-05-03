# Backend Plan — AI Poster Agent

> Goal: turn the current mock `/api/generate` into a real, observable, evaluated
> LangGraph agent that produces structured HTML/CSS posters. Built deliberately,
> step-by-step, so it can be explained end-to-end in an interview.

Each step has three sections:

- **What we're building** — the concrete deliverable.
- **Decisions to make** — the architectural choices we will pause on. We pick
  one explicitly, write down *why*, then implement.
- **Done when** — the checkable exit criteria.

We do **not** start a step until the previous one is "done." We do **not** skip
the decision discussions — that's the interview value.

---

## Target architecture (north star)

```
HTTP request                          LangSmith
   │                                      ▲
   ▼                                      │ traces
FastAPI (main.py)                         │
   │                                      │
   ▼                                      │
PosterAgent (agent.py)  ─── wraps ────►  agent_graph.py  (LangGraph)
                                          │
                                          ├── plan_node      (decompose brief)
                                          ├── design_node    (color/layout choices)
                                          ├── generate_node  (HTML + CSS, structured)
                                          ├── critique_node  (self-check)
                                          └── revise_node    (apply fixes)
                                               │
                                               ▼
                                          llm_client.py  (litellm → Gemini / Ollama)
                                               │
                                               ▼
                                          Pydantic structured output

                                          tools.py    (palette, font, image-search)

evals/ragas_eval.py  ──► offline scoring on a fixed set of briefs
```

Final API contract stays the same (`{html, css, session_id, ...}`) so the
frontend doesn't change.

---

## Step 0 — Lock the scope and contract  *(no code)*  ✅ DONE

**What we're building.** A short written agreement on:

- Inputs: `prompt: str`, `session_id: str | None`, `use_memory: bool`.
- Output schema: see Pydantic sketch below.
- Non-goals (for v1): image generation, real font assets, multi-page posters,
  authentication, persistence beyond in-memory sessions.

**Decisions made.**

1. **Single-shot vs. multi-turn agent.** ✅ **Single graph.**
   *Why:* sufficient for this project; one trace per request, easier to
   evaluate. Multi-graph deferred until a real requirement forces it.
   `use_memory` will flip a conditional edge inside the single graph.

2. **Where memory lives.** ✅ **Plain in-memory dict (current `SESSIONS`).**
   *Why:* keep what we have; don't add LangGraph `MemorySaver` yet. Easier to
   reason about for v1. Swap to `MemorySaver` (or Postgres checkpointer) in
   Step 9 if there's time, or when persistence becomes a real need.

### Response schema sketch

```python
# Returned by /api/generate (unchanged shape — frontend stays the same)
class GenerateResponse(BaseModel):
    session_id: str
    html: str
    css: str
    used_memory: bool
    history_length: int
    # New, optional fields the agent will populate (frontend can ignore for now):
    rationale: str | None = None        # 1-2 sentence design reasoning
    palette: list[str] | None = None    # hex codes the design used
```

The two new fields (`rationale`, `palette`) are additive and optional, so
adding them does not break the current frontend. We can surface them in the
UI later as an "About this poster" panel.

**Done when.** `plan.md` has these decisions written into it ✅, and the
response Pydantic model is sketched ✅. Ready for Step 1.

---

## Step 1 — `llm_client.py` upgrade  *(small)*  ✅ DONE

**What we're building.** A reusable function:

```python
def chat(messages: list[dict], *, response_model: type[BaseModel] | None = None,
         temperature: float = 0.4, tags: list[str] | None = None) -> BaseModel | str
```

Wraps `litellm.completion`, reads `LOCAL_MODEL` env, and (when `response_model`
is given) parses JSON output into a Pydantic model with one retry on
`ValidationError`.

**Decisions to make.**

1. **How to enforce structured output.** Three viable paths:
   - *`response_format={"type": "json_object"}`* via litellm — works with
     Gemini/OpenAI, not always with Ollama models.
   - *Instructor library* — clean Pydantic-first API, retries built-in, but
     another dep to defend in interview.
   - *Manual JSON-mode prompt + `model_validate_json` with retry* — most
     transparent, easy to explain.
   - **Recommend (default):** manual JSON-mode + Pydantic validate + 1 retry.
     We can mention Instructor as the "production" upgrade.

2. **Model choice per node.** All nodes use the same model, or do cheap nodes
   (plan/critique) use a smaller/local model?
   - **Recommend:** same model for v1 (Gemini 2.5 Flash). Note the optimization
     in the readme: "would split by node cost in production."

**Done when.** `llm_client.chat(...)` returns either a string or a validated
Pydantic instance, and is called from a tiny `__main__` smoke test.

---

## Step 2 — `tools.py`  *(optional but interview-worthy)*  ✅ DONE

**What we're building.** A handful of deterministic, side-effect-free helpers
the agent can call:

- `pick_palette(mood: str) -> list[str]` — returns 5 hex colors from a curated
  table (no network, deterministic).
- `pick_font_pair(style: str) -> {heading, body}` — same idea.
- `safe_text(text: str) -> str` — escapes HTML.
- (stretch) `search_unsplash(query: str) -> str` — returns a URL; gated behind
  an env flag so the demo doesn't need the API key.

**Decisions to make.**

1. **Tools-as-LangGraph-tools vs. plain functions.** Wiring them as
   `@tool`-decorated callables lets the LLM choose to invoke them; calling them
   deterministically from a node is simpler.
   - *@tool + ToolNode* → cool to demo, adds a "tool-calling" path to traces,
     but the LLM may pick the wrong one.
   - *Plain functions called from nodes* → predictable, easier to evaluate.
   - **Recommend:** plain functions for v1; if time allows, expose `pick_palette`
     as a real tool to demo the pattern.

**Done when.** `tools.py` exists with unit-test-style asserts in `__main__`.

---

## Step 3 — Pydantic schemas for graph state  *(no LLM yet)*  ✅ DONE

**What we're building.** All schemas in one place (top of `agent_graph.py` or a
new `schemas.py`):

```python
class PosterBrief(BaseModel):
    raw_prompt: str
    title: str
    audience: str | None
    style: str | None
    details: str | None

class DesignSpec(BaseModel):
    mood: str
    palette: list[str]            # hex codes
    font_pair: dict[str, str]
    layout: Literal["hero", "grid", "split", "minimal"]
    rationale: str

class PosterDraft(BaseModel):
    html: str
    css: str

class Critique(BaseModel):
    issues: list[str]
    must_fix: bool

class GraphState(TypedDict):
    brief: PosterBrief
    design: DesignSpec | None
    draft: PosterDraft | None
    critique: Critique | None
    revisions_left: int
    final: PosterDraft | None
    history: list[str]            # prior prompts when use_memory=True
```

**Decisions to make.**

1. **`TypedDict` vs `BaseModel` for graph state.** LangGraph supports both.
   - *TypedDict* → idiomatic LangGraph, mutates fields directly.
   - *BaseModel* → validation everywhere, more typing pain.
   - **Recommend:** TypedDict for the graph state, Pydantic for the node-level
     LLM outputs (best of both).

**Done when.** Schemas import cleanly and we can hand-construct a fake state.

---

## Step 4 — Build the LangGraph  *(`agent_graph.py`)*  ✅ DONE

**What we're building.** The graph wiring (no prompt engineering yet — use
placeholders that just echo). Nodes:

1. `parse_brief` — take `raw_prompt`, populate `PosterBrief`.
2. `design` — call LLM with `DesignSpec` schema; may call `pick_palette`.
3. `generate` — call LLM with `PosterDraft` schema (HTML+CSS).
4. `critique` — call LLM with `Critique` schema.
5. `revise` — if `must_fix and revisions_left > 0`, regenerate; else terminate.

Edges:

```
START → parse_brief → design → generate → critique
                                            │
                            ┌──── must_fix ─┴──── ok ──┐
                            ▼                          ▼
                         revise → generate           END
```

**Decisions to make.**

1. **Critique loop or no?** Adds latency and cost; greatly improves output.
   - *No critique* → ~1s faster, simpler trace.
   - *1 critique pass, max 1 revision* → measurable quality lift, bounded cost.
   - **Recommend:** 1 critique + at most 1 revision. Cap with `revisions_left`.

2. **Parse-brief: LLM or regex?** The frontend already sends a structured
   "Title: X | Audience: Y | …" string.
   - *Regex/split* → free, deterministic, but brittle if the frontend changes.
   - *LLM-based* → handles arbitrary text from advanced users.
   - **Recommend:** start regex, fall back to LLM if a field is missing.

3. **Conditional edges.** `add_conditional_edges` on the critique node — keep
   the routing function tiny and pure; no I/O inside.

**Done when.** `python -m backend.agent_graph` runs end-to-end with stubbed
LLM responses and prints the final state.

---

## Step 5 — `agent.py` wrapper

**What we're building.** A `PosterAgent` class with one method,
`run(prompt, session_id, use_memory) -> dict`. It:

- Loads the compiled graph once at import time.
- Resolves `thread_id` from `session_id` (creates a new one if absent).
- Invokes `graph.invoke(initial_state, config={"configurable": {"thread_id": ...}})`.
- Maps `GraphState.final` → API response dict.

**Decisions to make.**

1. **Sync vs async invocation.** FastAPI handles both; LangGraph supports both.
   - *Sync* → simpler stack traces, fine for low concurrency.
   - *Async* → needed if we add streaming via SSE/websockets later.
   - **Recommend:** sync for v1; add `arun` only when we add streaming.

2. **Streaming the poster as it's built?** Cool demo, but the iframe needs the
   final HTML+CSS together to render meaningfully.
   - **Recommend:** stream *status* events (`"designing"`, `"generating"`,
     `"critiquing"`) but ship final HTML/CSS in one go. Defer to v1.5.

**Done when.** `main.py`'s `/api/generate` calls `PosterAgent().run(...)`
instead of `_mock_poster()`. Frontend works unchanged.

---

## Step 6 — Prompt engineering pass  *(this is the big quality lever)*

**What we're building.** A `prompts.py` (or constants in each module) with:

- **System prompt per node**, each containing: role, hard rules, output schema
  reminder, 1–2 few-shot examples.
- A shared "design language" preamble (consistent voice across nodes).

**Decisions to make.**

1. **Few-shot vs. zero-shot.** Few-shots cost tokens but stabilize structured
   output dramatically.
   - **Recommend:** 1 few-shot per node (input → exact JSON we want).

2. **HTML constraints.** Do we let the LLM write any HTML/CSS, or constrain to
   a fixed skeleton it fills in?
   - *Free-form* → creative, but inconsistent and harder to evaluate.
   - *Skeleton with named slots* (`{title}`, `{cells}`, `{palette}`) → reliable
     output, easier to score.
   - **Recommend:** hybrid — fixed outer container + slots, but allow the LLM
     to author inner styling. Trade-off explained in interview as
     "guardrails vs. creativity."

3. **Critique rubric.** Free text vs. fixed checklist (contrast, alignment,
   readability, prompt fidelity).
   - **Recommend:** fixed 4-item checklist; deterministic to score later in
     RAGAS.

**Done when.** Side-by-side compare: same brief through v0 prompts vs. v1
prompts; v1 wins on a manual rubric. Save both in `prompts/_examples.md`.

---

## Step 7 — LangSmith observability

**What we're building.**

- `LANGSMITH_API_KEY` + `LANGSMITH_PROJECT` env vars.
- `LANGCHAIN_TRACING_V2=true`. (LangGraph traces auto-flow when env is set.)
- `tags=["node:design"]` etc. on each LLM call so the LangSmith UI groups them.
- A short README section showing where to view traces.

**Decisions to make.**

1. **Tracing scope.** Only LLM calls, or graph nodes too?
   - **Recommend:** both. LangGraph emits node-level spans natively when
     LangSmith is configured.

2. **Custom metadata.** Attach `session_id` and `use_memory` as metadata so the
   LangSmith UI can filter by them — extremely useful for the interview demo.

3. **PII / log policy.** The poster prompt is user-provided text. For an
   interview project this is fine, but call it out: in production we'd hash or
   redact session IDs before sending to LangSmith.

**Done when.** A real run shows up in LangSmith with one trace, all 4–5 spans,
and the `session_id` filterable.

---

## Step 8 — Evaluation with RAGAS  *(`evals/ragas_eval.py`)*

**What we're building.**

- A small dataset (`evals/dataset.jsonl`): ~15 briefs covering different
  audiences/styles, each with a *reference* expectation (e.g. "must mention
  audience word", "palette should match mood").
- A script that:
  1. Runs the agent on every brief.
  2. Constructs RAGAS samples — but adapted: classic RAG metrics (faithfulness,
     answer_relevancy, context_precision) need a "context"; ours has tool
     outputs (palette, font pair) as the context.
  3. Reports a CSV/markdown table.

**Decisions to make.**

1. **RAGAS metric fit.** RAGAS is RAG-centric. Mappings for our agent:
   - *Faithfulness* → "does the HTML/CSS use the palette returned by
     `pick_palette`?" Treat tool output as context, generated CSS as answer.
   - *Answer Relevancy* → "does the poster reference the brief's title and
     audience?"
   - *Context Precision* → less applicable; we can drop or replace with a
     custom "schema validity" metric.
   - **Recommend:** keep `faithfulness` + `answer_relevancy`, add custom
     metrics for `schema_validity`, `palette_coverage`, `prompt_fidelity`.
     This is honest in an interview: *"RAGAS isn't a perfect fit; here's
     what I kept and what I replaced and why."*

2. **LLM-as-judge model.** Use the same model we generate with, or a different
   one (anti-bias)?
   - **Recommend:** use a different model for judging (e.g. `gemini-flash` for
     gen, `gemini-pro` for judge) — call this out as a known best practice.

3. **Where evals run.** Local script for now; LangSmith Datasets + Evaluators
   later.
   - **Recommend:** local script; mention LangSmith as the next step.

**Done when.** `python evals/ragas_eval.py` prints a results table and writes
`evals/results-<timestamp>.md`. We commit one baseline result file.

---

## Step 9 — Persistence + cleanup

- Replace in-memory `SESSIONS` with LangGraph `MemorySaver` (or Postgres
  checkpointer using the existing `db_client.py`).
- Add a tiny test (`pytest`) for `agent.run` happy path.
- Update root `README.md` and `Backend/README.md` with run, eval, and trace
  instructions.

---

## Decision log  *(updated as we go)*

| # | Decision | Chosen | Why | Date |
|---|----------|--------|-----|------|
| 0.1 | Single graph vs. multi-graph for generate/follow-up | **Single graph** with conditional edge on `use_memory` | Sufficient for v1; one trace per request; revisit when a real requirement demands separate flows | 2026-05-03 |
| 0.2 | Session memory backend | **In-memory `SESSIONS` dict** (current) | Keep what works; defer `MemorySaver` / Postgres checkpointer until persistence is a real need | 2026-05-03 |
| 1.1 | Structured output mechanism | **Manual JSON-mode prompt + `model_validate_json` + 1 retry** | Most transparent; no extra deps; easy to explain in interview. Instructor mentioned as production upgrade. | 2026-05-03 |
| 1.2 | Per-node model choice | **Same model for all nodes** | Only one API key available (Gemini); also keeps v1 simple. Will note "split by node cost" as a production optimization. | 2026-05-03 |
| 2.1 | Tools wiring | **Single `pick_palette` exposed as a LangGraph `@tool`** | One tool is enough to demonstrate tool-usage; LangGraph `@tool` + `ToolNode` lets the LLM decide whether to call it (skips when user already specified a palette) — that's the interview story. | 2026-05-03 |
| 3.1 | Graph state vs. LLM output schemas | **`TypedDict` for graph state, Pydantic `BaseModel` for LLM-facing schemas**, all in `schemas.py` | Idiomatic LangGraph for state; Pydantic gives validation at every LLM boundary. | 2026-05-03 |
| 4.1 | First node — input gate | **`validate` node** that checks if the prompt is a genuine poster request; on refusal returns a styled HTML/CSS message and routes to END | Real users send "hi" or "add two numbers"; we need a graceful refusal path before the agent burns tokens on design/generate. Validation also extracts the `PosterBrief` in the same LLM call to save tokens. | 2026-05-03 |
| 4.2 | Tool-calling pattern | **LLM-driven tool plan via Pydantic** (`ToolPlan` schema) executed inside `design_node`; `pick_template` is **mandatory** (enforced by required schema field) and `pick_palette` is **optional** (`Optional[str]`, set only when user didn't specify colors) | Stays compatible with our litellm-based `llm_client` (no need for LangChain `bind_tools`). LLM still chooses *which* template + whether to call palette → preserves the agentic tool-usage story. Mandatory-ness is a schema invariant, not a runtime check. | 2026-05-03 |
| 4.3 | Critique + revise loop | **Implemented but commented out** in `agent_graph.py` | API token budget. Code path is in place so we can flip it on later without restructuring the graph. | 2026-05-03 |
| 4.4 | Brief parsing | **Done inside `validate_node`** (returned alongside `ValidationResult.brief`) | Single LLM call instead of separate validate + parse-brief nodes; no separate `parse_brief_node` needed. | 2026-05-03 |

---

## How we work each step

1. Re-read the step's "Decisions to make."
2. I lay out 2–3 options with tradeoffs.
3. You pick (or ask follow-ups).
4. I write the chosen option into the **Decision log** above with a one-line
   *why*.
5. I implement only that step.
6. We sanity-check it (smoke test, trace, or eval as relevant).
7. We move to the next step — never two at once.

When you're ready, say "go Step 0" and we'll start with scope + contract.
