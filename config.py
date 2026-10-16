"""Settings, loaded once from the environment.

Everything configurable lives here so no other module reads os.environ
directly — that way a missing setting fails loudly at import, not halfway
through a user's question.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when a required setting is missing."""


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    chat_model: str
    database_url: str
    sample_rows: int

    @property
    def masked_key(self) -> str:
        k = self.google_api_key
        return f"{k[:4]}…{k[-4:]}" if len(k) > 10 else "…"


def _build_database_url() -> str:
    """Prefer an explicit DATABASE_URL; otherwise assemble a MySQL one.

    quote_plus is essential: an unescaped '@' or ':' in the password splits
    the URI and everything after it is parsed as the hostname.
    """
    explicit = os.environ.get("DATABASE_URL", "").strip()
    if explicit:
        return explicit

    user = os.environ.get("DB_USER", "root")
    password = _required("MYSQL_PASSWORD")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "3306")
    name = os.environ.get("DB_NAME", "atliq_tshirts")

    return f"mysql+pymysql://{user}:{quote_plus(password)}@{host}:{port}/{name}"


def get_settings() -> Settings:
    return Settings(
        google_api_key=_required("GOOGLE_API_KEY"),
        chat_model=os.environ.get("CHAT_MODEL", "gemini-3.5-flash-lite"),
        database_url=_build_database_url(),
        sample_rows=int(os.environ.get("SAMPLE_ROWS_IN_TABLE_INFO", "3")),
    )
