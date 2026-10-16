"""Streamlit UI for the t-shirt sales assistant.

Run with:
    streamlit run main.py
"""

from __future__ import annotations

import streamlit as st

from config import ConfigError
from langchain_helper import SAMPLE_QUESTIONS, QuotaExceededError, SalesAssistant

st.set_page_config(page_title="T-Shirt Sales Assistant", page_icon="👕")

st.title("👕 T-Shirt Sales Assistant")
st.caption("Ask about stock, pricing and discounts in plain English.")


@st.cache_resource(show_spinner="Connecting to the database…")
def get_assistant() -> SalesAssistant:
    """Built once per session — reconnecting on every question would be slow."""
    return SalesAssistant()


try:
    assistant = get_assistant()
except ConfigError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Could not connect to the database: {exc}")
    st.info(
        "Check that MySQL is running (`systemctl status mysql`) and that the "
        "credentials in `.env` are correct."
    )
    st.stop()

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.subheader("Connection")
    st.write(f"**Dialect:** `{assistant.db.dialect}`")
    st.write(f"**Tables:** {', '.join(assistant.tables)}")
    st.write(f"**Model:** `{assistant.settings.chat_model}`")

    st.divider()
    st.subheader("Try a question")
    for i, sample in enumerate(SAMPLE_QUESTIONS):
        if st.button(sample, key=f"sample{i}", use_container_width=True):
            st.session_state.question = sample

    st.divider()
    st.caption(
        "Free tier allows 20 chat requests per day, per model. "
        "Change `CHAT_MODEL` in `.env` for a fresh allowance."
    )

# ------------------------------------------------------------------- main --
question = st.text_input(
    "Your question:",
    key="question",
    placeholder="How many white Levi's t-shirts are available?",
)

if question:
    try:
        with st.spinner("Writing SQL and querying…"):
            answer = assistant.ask(question)
    except QuotaExceededError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Something went wrong: {exc}")
    else:
        st.header("Answer")
        st.write(answer.text)

        if answer.queries:
            with st.expander(f"SQL used ({len(answer.queries)} quer"
                             f"{'y' if len(answer.queries) == 1 else 'ies'})"):
                for query, result in zip(answer.queries, answer.results):
                    st.code(query, language="sql")
                    st.text(result)
        else:
            st.caption("No SQL was run — the answer did not come from the database.")

        st.info(
            "Natural language is ambiguous. If a number looks wrong, open the "
            "SQL above — the query is usually where the misunderstanding is.",
            icon="💡",
        )
