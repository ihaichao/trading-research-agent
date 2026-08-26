"""Application settings.

All configuration comes from environment variables (or a local .env file) with
the ``TRA_`` prefix. Nothing is hardcoded and no secret ever enters the repo.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded once per process."""

    model_config = SettingsConfigDict(
        env_prefix="TRA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    llm_api_key: str = Field(default="", description="API key for an OpenAI-compatible endpoint")
    llm_base_url: str = Field(default="https://api.openai.com/v1")
    llm_model: str = Field(default="gpt-4.1-mini")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # --- SEC EDGAR ---
    # The SEC requires a descriptive User-Agent that includes a contact email,
    # and caps traffic at 10 requests/second.
    sec_user_agent: str = Field(default="trading-research-agent contact@example.com")
    sec_rate_limit_per_sec: int = Field(default=8, ge=1, le=10)

    # --- Search (M4+) ---
    tavily_api_key: str = Field(default="")

    # --- Misc ---
    cache_dir: Path = Field(default=Path(".cache"))
    log_level: str = Field(default="INFO")

    def require_llm(self) -> None:
        """Fail fast with an actionable message instead of a 401 deep in a call stack."""
        if not self.llm_api_key:
            raise RuntimeError(
                "TRA_LLM_API_KEY is not set. Copy .env.example to .env and fill it in."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
