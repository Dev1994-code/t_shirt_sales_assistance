"""The LangChain layer: an agent that answers questions by writing SQL.

Deliberately returns the SQL it ran alongside the answer. Natural-language
questions are often ambiguous ("the price of the inventory" could mean
SUM(price) or SUM(price * stock_quantity)), and the agent will pick one
silently. Showing the query is the only way to catch that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

import database
from config import Settings, get_settings


@dataclass
class Answer:
    """What the assistant produced for one question."""

    text: str
    queries: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)


class QuotaExceededError(RuntimeError):
    """The free-tier daily chat limit was reached."""


def _text_of(message) -> str:
    """Gemini 3.x returns .content as a list of blocks rather than a string."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


SYSTEM_PROMPT = (
    "You answer questions about a {dialect} t-shirt store database.\n\n"
    "Always call show_schema first to learn the tables and columns.\n"
    "Then write ONE SQL SELECT and run it with run_sql.\n"
    "Base your answer only on what the query returns — never guess.\n\n"
    "Notes on this schema:\n"
    "- Sizes are stored as 'XS', 'S', 'M', 'L', 'XL'.\n"
    "- 'inventory value' means SUM(price * stock_quantity), not SUM(price).\n\n"
    "Discounts — read carefully:\n"
    "- Apply discounts ONLY when the question explicitly mentions discounts,\n"
    "  or asks for revenue 'after'/'post' discount. Otherwise ignore the\n"
    "  discounts table entirely.\n"
    "- When discounts ARE requested: pct_discount is a percentage, so the\n"
    "  figure is price * stock_quantity * (1 - pct_discount/100). Not every\n"
    "  t-shirt has one, so LEFT JOIN and treat a missing discount as zero.\n\n"
    "Answer in one or two short sentences, including the number."
)


class SalesAssistant:
    """Wraps the database, the model and the agent loop."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        os.environ["GOOGLE_API_KEY"] = self.settings.google_api_key

        self.db = database.connect(self.settings)
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.chat_model,
            temperature=0,  # deterministic: the same question -> the same SQL
            max_retries=3,
        )
        self._last_queries: list[str] = []
        self._last_results: list[str] = []
        self.agent = create_agent(
            self.llm,
            self._build_tools(),
            system_prompt=SYSTEM_PROMPT.format(dialect=self.db.dialect),
        )

    # ----------------------------------------------------------------- tools
    def _build_tools(self):
        db = self.db

        @tool
        def show_schema() -> str:
            """Return CREATE TABLE statements and sample rows for every table."""
            return db.table_info

        @tool
        def run_sql(query: str) -> str:
            """Run one read-only SQL query and return its rows."""
            self._last_queries.append(query.strip())
            try:
                result = database.run_read_only(db, query)
            except database.WriteAttemptError as exc:
                result = f"REFUSED: {exc}"
            except Exception as exc:  # malformed SQL — let the agent retry
                result = f"SQL error: {exc}"
            self._last_results.append(result)
            return result

        return [show_schema, run_sql]

    # ------------------------------------------------------------------- api
    @property
    def tables(self) -> list[str]:
        return self.db.get_usable_table_names()

    def ask(self, question: str) -> Answer:
        """Answer a natural-language question, reporting the SQL used."""
        self._last_queries = []
        self._last_results = []

        try:
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
        except Exception as exc:
            message = str(exc)
            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                raise QuotaExceededError(
                    "Daily free-tier chat quota reached (20 requests/day per "
                    "model). Wait for the reset, or set CHAT_MODEL in .env to "
                    "a different model — the quota is per model."
                ) from exc
            raise

        return Answer(
            text=_text_of(response["messages"][-1]),
            queries=list(self._last_queries),
            results=list(self._last_results),
        )


SAMPLE_QUESTIONS = [
    "How many t-shirts do we have left for Nike in extra small size and white color?",
    "What is the total inventory value for all small size t-shirts?",
    "If we sell all Levi's t-shirts today with discounts applied, how much revenue will we generate?",
    "How many white Levi's t-shirts are available?",
    "Which brand has the highest total stock?",
    "What is the average price per brand?",
]
