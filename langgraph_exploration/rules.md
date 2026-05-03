# Production rules of thumb — LangGraph, LangSmith, Structured Outputs

A consolidated checklist from this exploration. Each rule has a one-line
*why* — that's the part you actually need to remember; the rule itself is a
consequence.

---

## LangGraph

### Graph design
1. **One `.invoke()` = one trace tree.** Don't try to share state across
   invocations through globals; pass it through the state dict or use
   `MemorySaver` keyed by `thread_id`. *Why:* invocations are the unit of
   observability, replay, and persistence.
2. **Nodes are pure functions over a state snapshot.** No mutating module
   globals, no I/O on shared resources without isolation. *Why:* LangGraph
   may run nodes in parallel, retry them, or replay them from a checkpoint —
   side effects break all three.
3. **Different keys for parallel writers; reducers for shared keys.** If two
   parallel nodes append to the same list, type that field as
   `Annotated[list, operator.add]`. *Why:* without a reducer, concurrent
   writes to the same key raise an `InvalidUpdateError`.
4. **Keep conditional-edge functions tiny and pure.** No LLM calls, no DB
   reads — just route. *Why:* routing logic is replayed on every checkpoint
   resume; expensive routing makes resumes expensive.
5. **Bound your loops.** Any `revise → generate → critique` cycle needs an
   explicit counter (`revisions_left`) in state and a hard cap. *Why:* a
   self-critiquing loop will happily burn $50 on one request if nothing
   stops it.
6. **Name nodes for what they *do*, not what they *are*.** `parse_brief`
   beats `node_1`; `analyze_word_1` beats `essay_node`. *Why:* node names
   become span names in LangSmith — your future self reading a 30-span
   trace will thank you.

### State and types
7. **`TypedDict` for graph state, `BaseModel` for node-level LLM outputs.**
   *Why:* TypedDict is what LangGraph reducers expect; Pydantic is what you
   need at the LLM boundary. Mixing them confuses both.
8. **Initialize every state field at invocation.** `None` for objects,
   `[]` for lists, `""` for strings. *Why:* unset fields surface as
   `KeyError` deep inside a node, far from the cause.
9. **Treat state as your contract.** Adding a required field is a breaking
   change for in-flight checkpoints. *Why:* `MemorySaver` deserializes old
   state shapes; a new required field invalidates them.

### Async, streaming, persistence
10. **Sync until you have a streaming need.** `arun` adds real complexity
    (context propagation, cancellation, exception traces). *Why:* premature
    async is the most common LangGraph footgun.
11. **`MemorySaver` for demos, durable checkpointer (Postgres/SQLite) for
    anything real.** *Why:* in-memory state evaporates on restart; users
    lose mid-conversation work and you lose your replay capability.
12. **Stream status events, not partial structured outputs.** `"designing"`,
    `"generating"`, `"critiquing"` — yes. Streaming a half-formed JSON
    object — no. *Why:* the consumer can't render half a Pydantic model;
    they can render a status string.

---

## LangSmith

### Setup hygiene
13. **`LANGSMITH_*` env vars over `LANGCHAIN_*`.** Both work; the former is
    canonical. *Why:* future docs and migrations align with the `LANGSMITH_`
    prefix.
14. **One project per logical agent, not per environment.** Use `tags` or
    `metadata.environment=staging|prod` for dev/staging/prod separation
    inside the same project. *Why:* you compare *the same agent* across
    environments far more often than you compare *different agents*.
15. **Different projects for genuinely different agents.** Set
    `project_name` per `.invoke()` config. *Why:* one noisy agent shouldn't
    pollute another's metrics dashboards.

### Tracing scope
16. **Set `run_name` on every `.invoke()`.** The default `"LangGraph"` label
    is useless when you have ten graphs. *Why:* span names are how you find
    things in the UI; "LangGraph" is the same as "untitled."
17. **Use `tags` for *kinds* of runs, `metadata` for *values*.** Tag
    `["essay-demo", "v2-prompts"]`, metadata `{"user_id": "...",
    "word1": "..."}`. *Why:* tags are filterable enums in the UI; metadata
    is searchable text. Mixing them collapses both UX paths.
18. **`@traceable` only what's worth a span.** Decorate LLM calls, tool
    calls, retrievers. Don't decorate every helper. *Why:* spans cost you
    nothing in $ but cost you everything in trace readability.
19. **Pick the right `run_type`.** `"llm"`, `"tool"`, `"retriever"`,
    `"chain"`, `"prompt"`, `"parser"`, `"embedding"`. *Why:* the UI renders
    each type differently — a retriever shows "documents," an LLM shows
    token usage. The wrong type loses you all of that.

### Privacy and cost
20. **Hash or redact PII before it leaves your process.** Session IDs,
    emails, anything regulated. *Why:* LangSmith stores everything you send;
    a leak there is a leak from your system.
21. **Sample in prod, full-trace in dev.** Use
    `LANGSMITH_SAMPLING_RATE=0.1` for high-volume endpoints. *Why:*
    LangSmith is metered; 100% of a 1k-RPS endpoint will surprise you on
    invoice day.
22. **Treat traces as production logs, not debug logs.** Don't log secrets,
    raw user emails, or full system prompts you consider proprietary.
    *Why:* anyone with project access reads them.

### Datasets and evaluation
23. **Promote real traces to datasets, don't synthesize them.** Click
    "Add to Dataset" on traces that surprised you (good or bad). *Why:*
    synthetic eval sets test what you imagined, not what users do.
24. **Use a different judge model than your generator.** Gemini Flash gens,
    Gemini Pro judges (or vice versa, or a different vendor entirely).
    *Why:* same-model judging has measurable self-preference bias.
25. **Pin eval set version + judge model + rubric in your eval results.**
    *Why:* a "score went up" comparison is meaningless if any of those
    three changed silently.

---

## Structured outputs

### Validation discipline
26. **Always validate with Pydantic at the boundary.** Even with native
    structured output, even with tool calling. *Why:* providers occasionally
    relax constraints, models occasionally drift. Defense in depth is cheap.
27. **Always have one repair retry, with the error fed back.** Send the
    `ValidationError` text and the previous raw output back to the model.
    *Why:* this single retry recovers ~90% of structured-output failures
    for free.
28. **Cap retries at 1, maybe 2.** Beyond that the model is wrong in a way
    retries won't fix; surface the error. *Why:* retry storms inflate
    latency and cost without improving success rate.
29. **Treat truncation as a hard error.** Check `finish_reason == "length"`
    and raise. *Why:* truncated JSON is the #1 silent cause of structured
    output failures; do *not* try to repair-parse it.

### Schema design
30. **Generate the schema from Pydantic; never hand-write it.** Use
    `Model.model_json_schema()` and inject into the prompt. *Why:* hand-
    written schemas drift the moment someone refactors the model.
31. **Field descriptions are prompt-engineering, not docstrings.** The model
    reads `Field(description=...)`. Write for the model, not the IDE
    tooltip. *Why:* this is the cheapest, highest-leverage place to
    influence output quality.
32. **Self-describing field names.** `origin_of_word` beats `origin` beats
    `o1`. *Why:* names are tokens the model sees and is biased by.
33. **Constrain with Pydantic where you can.** `min_length`, `max_length`,
    `pattern`, `Literal[...]`, `conint(ge=0, le=100)`. *Why:* validators
    catch errors the model can't even see; "must be 3-5 items" enforced
    silently is better than "please give me 3-5 items" hoped for.
34. **Optional fields with defaults for additive changes.** Adding a
    required field is breaking; adding `Optional[...] = None` is not.
    *Why:* downstream consumers (and stored historical traces) survive
    schema evolution.

### Prompting and parameters
35. **Lower the temperature for shape-sensitive outputs.** `0.0–0.2` for
    structured envelopes; the *content* inside a free-text field can still
    be at higher temp via prompting. *Why:* temperature variance breaks
    schemas before it breaks prose.
36. **Strip markdown fences defensively.** Models love wrapping JSON in
    ` ```json ... ``` ` even when forbidden. *Why:* one regex in your
    sanitizer prevents a class of false-positive validation errors.
37. **Set `max_tokens` generously and explicitly.** Default limits are
    optimized for chat, not for nested JSON. *Why:* you'd rather pay for
    headroom than debug truncation.
38. **Inject the JSON schema and explicit "JSON only" rules in the system
    prompt.** Both. Together. *Why:* JSON mode guarantees parseable JSON,
    not *correct* JSON; the prompt is what makes it correct.

### Picking a mechanism
39. **Default to: JSON mode + Pydantic + retry.** Works on every provider,
    transparent, easy to debug. *Why:* most portable for the least
    cognitive load.
40. **Upgrade to native structured output (OpenAI / Gemini) when you can
    pin the provider.** It's grammar-constrained at sample time and never
    fails schema. *Why:* you trade portability for correctness; on a
    locked stack, take it.
41. **Reach for tool/function calling when you also need real tools.** Same
    call, one less mode-switch. *Why:* mixing JSON mode and tool calls in
    the same agent is two mental models for one outcome.
42. **Reach for Instructor / `with_structured_output()` when maintaining
    the dispatch yourself stops being interesting.** *Why:* the wrapper is
    a dependency, but it's a smaller one than the bug surface of a
    hand-rolled multi-provider abstraction.

### Operational
43. **Log raw output on every validation failure.** Span input/output via
    LangSmith covers this for free. *Why:* you cannot debug what you did
    not capture.
44. **Don't trust LLM-emitted IDs, foreign keys, or numerics that need
    referential integrity.** Validate against the source of truth before
    use. *Why:* hallucinated IDs are a real outage class — payments routed
    to non-existent accounts, queries against non-existent tables.
45. **Treat your output schema as a public API.** Version it; document
    breaking changes; communicate them. *Why:* anyone consuming your
    structured output (frontend, downstream service, eval pipeline) breaks
    silently when you rename a field.

---

## Meta

46. **Trace first, then optimize.** Don't tune prompts you can't observe.
    *Why:* every prompt change without a trace is a guess.
47. **Validate first, then trust.** Don't build downstream logic on the
    assumption the LLM "usually" returns the right shape. *Why:* "usually"
    is exactly the failure mode that survives staging and dies in prod.
48. **One change at a time.** New model, new prompt, new schema, new graph
    structure — never two together. *Why:* you can't attribute a regression
    to a change you didn't isolate.
