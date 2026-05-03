"""
Streamlit UI for the parallel structured-analysis LangGraph.

Two word inputs → graph runs both analyses in parallel → renders three
sections per word (origin, essay, usage). Errors are caught and shown
inline so structured-output failures are visible, not silent.

Run with:
    streamlit run app.py
"""

import streamlit as st

from workflow import run

st.set_page_config(page_title="LangGraph parallel word analysis", page_icon=None)
st.title("LangGraph parallel word analysis")
st.caption(
    "Two words → two parallel LangGraph nodes → each returns a validated, "
    "three-section Pydantic object → combined response."
)

with st.form("inputs"):
    col1, col2 = st.columns(2)
    with col1:
        word1 = st.text_input("Word 1", value="serendipity")
    with col2:
        word2 = st.text_input("Word 2", value="ephemeral")
    submitted = st.form_submit_button("Analyze")


def render_word_block(term: str, analysis: dict) -> None:
    st.subheader(term)
    st.markdown("**Origin**")
    st.write(analysis["first"]["origin_of_word"])
    st.markdown("**Essay**")
    st.write(analysis["second"]["essay"])
    st.markdown("**Usage**")
    for example in analysis["third"]["word_usage"]:
        st.markdown(f"- {example}")


if submitted:
    if not word1.strip() or not word2.strip():
        st.error("Both words are required.")
    else:
        with st.spinner("Running the graph..."):
            try:
                output = run(word1.strip(), word2.strip())
                st.success("Done.")
                left, right = st.columns(2)
                with left:
                    render_word_block(
                        output["word1"]["term"], output["word1"]["analysis"]
                    )
                with right:
                    render_word_block(
                        output["word2"]["term"], output["word2"]["analysis"]
                    )
                with st.expander("Raw combined JSON"):
                    st.json(output)
            except Exception as e:
                st.error(f"Workflow failed: {type(e).__name__}: {e}")
                st.exception(e)
