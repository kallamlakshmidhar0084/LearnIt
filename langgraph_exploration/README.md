# LangGraph + LangSmith exploration

A tiny, end-to-end demo of:

- A **LangGraph** workflow with two **parallel** nodes that fan out from `START`
  and merge at a `combine` node.
- Reusing the existing `llm_client.py` (LiteLLM → Gemini / Ollama) for all LLM
  calls.
- A minimal **Streamlit** UI to drive the graph.
- **LangSmith** tracing wired up so every node + LLM call shows up as spans in
  one trace.

## What's in here

| File | Purpose |
|------|---------|
| [llm_client.py](llm_client.py) | Copy of `Backend/llm_client.py`, adapted to expose a `chat(prompt, ...)` function with a `@traceable` decorator. |
| [workflow.py](workflow.py) | The LangGraph: `START → {write_essay_1, write_essay_2} → combine → END`. |
| [app.py](app.py) | Streamlit UI with two word inputs, a submit button, and pass/fail rendering. |
| [requirements.txt](requirements.txt) | langgraph, langsmith, litellm, python-dotenv, streamlit. |
| [.env.example](.env.example) | Template for `GEMINI_API_KEY` + LangSmith vars. |

## Graph

```
           ┌──► write_essay_1 ──┐
    START ─┤                    ├──► combine ──► END
           └──► write_essay_2 ──┘
```

`write_essay_1` and `write_essay_2` are scheduled concurrently because both
have `START` as their only predecessor and they write to *different* state
keys (`essay1` / `essay2`), so no reducer is needed.

## Run it

```bash
cd langgraph_exploration
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in real keys
streamlit run app.py
```

Or run the graph directly without UI:

```bash
python workflow.py
```

---

## LangSmith setup — step by step

LangSmith is the observability/eval platform from the LangChain team.
LangGraph emits traces to it automatically as long as the right env vars are
set — no code changes required for graph-level tracing.

### 1. Create a LangSmith account
- Go to https://smith.langchain.com and sign up (Google/GitHub SSO works).
- It's free for individual use up to a generous trace volume.

### 2. Create a project
- In the left sidebar click **Projects → + New Project**.
- Name it `langgraph-exploration` (matches the value in `.env.example`).
- This is the bucket all traces from this app will land in.

### 3. Create an API key
- Top-right avatar → **Settings → API Keys → Create API Key**.
- Give it a name like `local-dev` and copy the key (`lsv2_pt_...`).
- You won't be able to see it again, so paste it into `.env` immediately.

### 4. Set environment variables
Copy `.env.example` to `.env` and fill in:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxx
LANGSMITH_PROJECT=langgraph-exploration
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

> Older docs use `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`.
> Both prefixes work; the `LANGSMITH_*` form is the newer canonical one.

`python-dotenv` (already imported in `llm_client.py`) loads `.env` on import,
so just running `streamlit run app.py` is enough.

### 5. Verify tracing
- Run the UI, submit two words, wait for the response.
- Open https://smith.langchain.com → **Projects → langgraph-exploration**.
- You should see one new trace per submission. Open it and you'll see:
  - A root span for the graph invocation.
  - Two parallel child spans: `write_essay_1` and `write_essay_2` (note their
    overlapping start/end times — that's the parallelism).
  - Each essay node has a nested `llm_client.chat` LLM span (added by the
    `@traceable` decorator on `chat()`).
  - A final `combine` span before END.

### 6. (Optional) Add metadata / tags
For richer filtering inside LangSmith, you can pass metadata when invoking the
graph:

```python
GRAPH.invoke(
    state,
    config={
        "tags": ["demo", "parallel-essays"],
        "metadata": {"word1": word1, "word2": word2},
    },
)
```

Then in the LangSmith UI you can filter the trace list by `tags` or by any
metadata key. Useful when you start running evaluations.

### 7. (Optional) Datasets + evaluators
Once you have traces, click **Add to Dataset** on any trace to start building
a regression set. From there you can run evaluators (LLM-as-judge or custom
Python) against the dataset on every change. That's the natural next step
after this demo.

---

## How the parallelism actually works

LangGraph builds a DAG from your edges and walks it in topological layers.
On each "super-step" it runs every node whose dependencies are satisfied —
in parallel, in threads. Because `write_essay_1` and `write_essay_2` both
depend only on `START`, they're in the same layer and run concurrently.
`combine` is in the next layer because it depends on *both* — LangGraph
waits for both to finish before invoking it.

Two things to remember if you extend this:

1. **Different keys = no reducer needed.** Both nodes return dicts with
   different keys (`essay1` vs `essay2`), so their state updates don't
   collide. If you wanted both to append to the *same* list (e.g. an `essays`
   list), you'd type that field as `Annotated[list[str], operator.add]` so
   LangGraph knows how to merge concurrent writes.
2. **Don't share mutable state across nodes.** Treat each node as a pure
   function over the snapshot it receives.
