"""
LangGraph workflow: parallel structured-analysis demo.

Graph shape:

           ┌──► analyze_word_1 ──┐
    START ─┤                     ├──► combine ──► END
           └──► analyze_word_2 ──┘

Each `analyze_word_*` node calls `chat_structured()` and gets back a fully
validated `WordAnalysis` Pydantic instance with three sections:
  - first.origin_of_word
  - second.essay
  - third.word_usage  (list of example sentences)

The combine node packages both into the final output.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from llm_client import chat_structured
from schemas import WordAnalysis


class EssayState(TypedDict):
    word1: str
    word2: str
    analysis1: WordAnalysis | None
    analysis2: WordAnalysis | None
    combined: dict


ANALYSIS_SYSTEM = (
    "You are a meticulous etymologist and essayist. For the word the user "
    "gives you, produce a three-section structured analysis: its origin, a "
    "200-word essay, and example usages."
)


def _analyze(word: str) -> WordAnalysis:
    return chat_structured(
        f"Analyze the word: '{word}'.",
        response_model=WordAnalysis,
        system=ANALYSIS_SYSTEM,
        temperature=0.4,
    )


def analyze_word_1(state: EssayState) -> dict:
    return {"analysis1": _analyze(state["word1"])}


def analyze_word_2(state: EssayState) -> dict:
    return {"analysis2": _analyze(state["word2"])}


def combine(state: EssayState) -> dict:
    combined = {
        "word1": {
            "term": state["word1"],
            "analysis": state["analysis1"].model_dump(),
        },
        "word2": {
            "term": state["word2"],
            "analysis": state["analysis2"].model_dump(),
        },
    }
    return {"combined": combined}


def build_graph():
    builder = StateGraph(EssayState)
    builder.add_node("analyze_word_1", analyze_word_1)
    builder.add_node("analyze_word_2", analyze_word_2)
    builder.add_node("combine", combine)

    builder.add_edge(START, "analyze_word_1")
    builder.add_edge(START, "analyze_word_2")
    builder.add_edge("analyze_word_1", "combine")
    builder.add_edge("analyze_word_2", "combine")
    builder.add_edge("combine", END)

    return builder.compile().with_config(run_name="parallel-word-analysis")


GRAPH = build_graph()


def run(word1: str, word2: str) -> dict:
    result = GRAPH.invoke(
        {
            "word1": word1,
            "word2": word2,
            "analysis1": None,
            "analysis2": None,
            "combined": {},
        },
        config={
            "tags": ["essay-demo", "structured"],
            "metadata": {"word1": word1, "word2": word2},
        },
    )
    return result["combined"]


if __name__ == "__main__":
    import json

    print(json.dumps(run("serendipity", "ephemeral"), indent=2))
