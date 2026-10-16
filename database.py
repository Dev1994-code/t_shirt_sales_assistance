"""Database access, wrapped so the rest of the app never builds SQL itself."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")  # langchain-community sunset notice

from langchain_community.utilities import SQLDatabase  # noqa: E402

from config import Settings

# Statements the assistant is allowed to run. Anything else is refused before
# it reaches the database — the model never gets a write path.
READ_ONLY_PREFIXES = ("select", "with", "show", "describe", "desc", "explain")


class WriteAttemptError(RuntimeError):
    """Raised when a non-read-only statement is attempted."""


def connect(settings: Settings) -> SQLDatabase:
    return SQLDatabase.from_uri(
        settings.database_url,
        sample_rows_in_table_info=settings.sample_rows,
    )


def is_read_only(query: str) -> bool:
    cleaned = query.strip().rstrip(";").lstrip("(").lower()
    return cleaned.startswith(READ_ONLY_PREFIXES)


def run_read_only(db: SQLDatabase, query: str) -> str:
    """Execute a query after confirming it only reads."""
    if not is_read_only(query):
        raise WriteAttemptError(
            "Only SELECT/WITH/SHOW/DESCRIBE/EXPLAIN statements are permitted."
        )
    return db.run(query.strip().rstrip(";")) or "(no rows)"
