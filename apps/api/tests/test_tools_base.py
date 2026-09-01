"""工具外壳的测试：契约、缓存、限速。"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from tra.config import get_settings
from tra.report.schema import MetricPoint, MetricSeries, Source, SourceKind, make_source_id
from tra.tools import RateLimiter, ToolResult, clear_cache, disk_cache

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """每个测试用自己的缓存目录，互不污染，也不碰开发者本地的 .cache/。

    get_settings 带 lru_cache，改完环境变量必须手动清掉，否则读到旧配置。
    """
    monkeypatch.setenv("TRA_CACHE_DIR", str(tmp_path / "cache"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _source() -> Source:
    return Source(
        id=make_source_id("https://example.com/10q", "Item 2"),
        kind=SourceKind.SEC_FILING,
        title="Example 10-Q",
        url="https://example.com/10q",
        locator="10-Q FY2026Q2, Item 2",
        retrieved_at=NOW,
    )


# --------------------------------------------------------------------- 契约


def test_success_carries_data_and_sources() -> None:
    src = _source()
    series = MetricSeries(
        key="revenue",
        label="Total revenue",
        unit="USD_millions",
        points=[MetricPoint(period="FY2026Q2", value=46743.0)],
        source_ids=[src.id],
    )
    result = ToolResult[list[MetricSeries]].success([series], [src])

    assert result.ok
    assert result.data is not None
    assert result.data[0].key == "revenue"
    assert result.sources == [src]
    assert result.error is None


def test_for_model_renders_data_and_sources() -> None:
    src = _source()
    result = ToolResult[str].success("hello", [src])
    text = result.for_model()

    assert "hello" in text
    assert src.id in text
    assert "10-Q FY2026Q2, Item 2" in text


def test_failure_renders_error_and_hint() -> None:
    """错误信息必须是可执行的：光说错了没用，要说下一步能干什么。"""
    result = ToolResult[float].failure(
        "tool 'get_quotes' not found",
        hint="available tools: get_quote, get_financials",
    )
    text = result.for_model()

    assert not result.ok
    assert "ERROR" in text and "get_quotes" in text
    assert "HINT" in text and "get_financials" in text


def test_from_exception_keeps_the_exception_type() -> None:
    """类型名让模型能区分该重试还是该换参数。"""
    result = ToolResult[float].from_exception(TimeoutError("read timed out"))
    assert "TimeoutError" in (result.error or "")
    assert "read timed out" in (result.error or "")


# --------------------------------------------------------------------- 缓存


def test_cache_avoids_a_second_call() -> None:
    calls = {"n": 0}

    @disk_cache(namespace="t", ttl=timedelta(days=1))
    def fetch(x: int) -> dict[str, int]:
        calls["n"] += 1
        return {"value": x * 2}

    assert fetch(21) == {"value": 42}
    assert fetch(21) == {"value": 42}
    assert calls["n"] == 1, "第二次应该命中缓存，不该再调用原函数"


def test_cache_key_includes_the_arguments() -> None:
    calls = {"n": 0}

    @disk_cache(namespace="t", ttl=None)
    def fetch(x: int) -> int:
        calls["n"] += 1
        return x

    fetch(1)
    fetch(2)
    assert calls["n"] == 2, "不同参数必须是不同的缓存键"


def test_expired_cache_is_refetched() -> None:
    calls = {"n": 0}

    @disk_cache(namespace="t", ttl=timedelta(seconds=0))
    def fetch() -> int:
        calls["n"] += 1
        return calls["n"]

    fetch()
    fetch()
    assert calls["n"] == 2, "ttl=0 意味着永远过期"


def test_corrupt_cache_file_is_ignored() -> None:
    """缓存文件损坏时应该重新取，而不是让整个流程炸掉。"""
    calls = {"n": 0}

    @disk_cache(namespace="t", ttl=None)
    def fetch() -> int:
        calls["n"] += 1
        return 7

    assert fetch() == 7

    from tra.tools.cache import cache_root

    for path in (cache_root() / "t").glob("*.json"):
        path.write_text("{ 这不是合法 JSON", encoding="utf-8")

    assert fetch() == 7
    assert calls["n"] == 2


def test_clear_cache_removes_the_namespace() -> None:
    calls = {"n": 0}

    @disk_cache(namespace="t", ttl=None)
    def fetch() -> int:
        calls["n"] += 1
        return 1

    fetch()
    clear_cache("t")
    fetch()
    assert calls["n"] == 2


def test_decorator_keeps_the_original_name() -> None:
    @disk_cache(namespace="t", ttl=None)
    def my_special_fetch() -> int:
        """原始 docstring。"""
        return 1

    assert my_special_fetch.__name__ == "my_special_fetch"
    assert my_special_fetch.__doc__ == "原始 docstring。"


# --------------------------------------------------------------------- 限速


def test_burst_passes_immediately() -> None:
    limiter = RateLimiter(rate_per_sec=100, burst=5)
    start = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    assert time.monotonic() - start < 0.05, "桶里有令牌时不该等待"


def test_sustained_rate_is_throttled() -> None:
    """突发额度用完之后，速率必须被压到设定值。"""
    limiter = RateLimiter(rate_per_sec=50, burst=1)

    start = time.monotonic()
    for _ in range(6):
        limiter.acquire()
    elapsed = time.monotonic() - start

    # 第一个走突发额度，剩下 5 个每个要等 1/50 秒
    assert elapsed >= 5 / 50 * 0.8, f"限速没生效，只用了 {elapsed:.3f}s"


def test_rejects_impossible_request() -> None:
    limiter = RateLimiter(rate_per_sec=10, burst=2)
    with pytest.raises(ValueError, match="最多取"):
        limiter.acquire(5)


def test_rejects_bad_rate() -> None:
    with pytest.raises(ValueError, match="必须大于 0"):
        RateLimiter(rate_per_sec=0)


def test_sec_limiter_is_a_shared_singleton() -> None:
    """并行 researcher 共用一个限速器，否则 N 个并发就是 N 倍速率。"""
    from tra.tools import sec_limiter

    assert sec_limiter() is sec_limiter()
