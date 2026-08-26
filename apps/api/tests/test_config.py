"""Settings should read from the environment and fail loudly when a key is missing."""

from __future__ import annotations

import pytest

from tra.config import Settings


def test_reads_env_with_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRA_LLM_MODEL", "some-model")
    settings = Settings(_env_file=None)
    assert settings.llm_model == "some-model"


def test_defaults_are_sane() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_base_url.startswith("http")
    assert 1 <= settings.sec_rate_limit_per_sec <= 10


def test_require_llm_raises_without_key() -> None:
    settings = Settings(_env_file=None, llm_api_key="")
    with pytest.raises(RuntimeError, match="TRA_LLM_API_KEY"):
        settings.require_llm()


def test_require_llm_passes_with_key() -> None:
    settings = Settings(_env_file=None, llm_api_key="sk-test")
    settings.require_llm()
