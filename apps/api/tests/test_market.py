"""行情工具的测试。不联网——用假 provider 驱动降级链路。

重点不是"能不能取到价格"，而是**取不到时会发生什么**。
yfinance 会挂，这是已知事实；工具的价值在于挂了之后不让模型编数字。
"""

from __future__ import annotations

from typing import Any

import pytest

from tra.config import get_settings
from tra.tools import market


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TRA_CACHE_DIR", str(tmp_path / "cache"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeProvider:
    def __init__(self, name: str, *, price: float | None = None, fail: str | None = None) -> None:
        self.name = name
        self.price = price
        self.fail = fail
        self.calls = 0

    def quote(self, ticker: str) -> dict[str, Any]:
        self.calls += 1
        if self.fail:
            raise RuntimeError(self.fail)
        return {"price": self.price, "currency": "USD", "as_of": "2026-08-26T00:00:00+00:00"}

    def history(self, ticker: str, *, period: str, interval: str) -> list[dict[str, Any]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError(self.fail)
        return [
            {"date": "2026-07-31", "close": 180.0},
            {"date": "2026-08-26", "close": self.price},
        ]


def use(monkeypatch: pytest.MonkeyPatch, *providers: FakeProvider) -> None:
    monkeypatch.setattr(market, "PROVIDERS", list(providers))


def test_quote_returns_price_and_a_market_data_source(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeProvider("primary", price=184.3))
    result = market.get_quote("nvda")

    assert result.ok
    assert result.data is not None
    assert result.data.ticker == "NVDA"
    assert result.data.price == pytest.approx(184.3)
    assert result.sources[0].kind == "market_data"
    assert "delayed" in result.sources[0].snippet


def test_falls_back_to_the_next_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = FakeProvider("broken", fail="Yahoo returned 429")
    backup = FakeProvider("backup", price=99.5)
    use(monkeypatch, broken, backup)

    result = market.get_quote("NVDA")

    assert result.ok
    assert result.data is not None
    assert result.data.provider == "backup"
    assert broken.calls == 1, "坏掉的源应该被试过一次再放弃"


def test_all_providers_down_tells_the_model_not_to_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """这条是这个文件里最重要的测试。

    行情取不到时，最糟的结果不是"没数据"，是模型顺口编一个价格。
    hint 必须明确禁止它这么做。
    """
    use(monkeypatch, FakeProvider("a", fail="timeout"), FakeProvider("b", fail="429"))
    result = market.get_quote("NVDA")

    assert not result.ok
    assert "timeout" in (result.error or "") and "429" in (result.error or "")
    assert "do not estimate" in (result.hint or "").lower()


def test_history_becomes_a_metric_series(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeProvider("primary", price=184.3))
    result = market.get_price_history("NVDA", period="1y", interval="1mo")

    assert result.ok
    assert result.data is not None
    assert result.data.key == "close_price"
    assert result.data.unit == "USD"
    assert [p.period for p in result.data.points] == ["2026-07-31", "2026-08-26"]
    assert result.data.source_ids == [result.sources[0].id]


def test_second_call_is_served_from_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """缓存不只是省时间——数据源挂掉时，它是最后一道防线。"""
    provider = FakeProvider("primary", price=184.3)
    use(monkeypatch, provider)

    market.get_quote("NVDA")
    market.get_quote("NVDA")

    assert provider.calls == 1


def test_different_tickers_are_cached_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeProvider("primary", price=1.0)
    use(monkeypatch, provider)

    market.get_quote("NVDA")
    market.get_quote("AMD")

    assert provider.calls == 2


@pytest.mark.network
def test_real_yfinance_quote() -> None:
    result = market.get_quote("NVDA")
    assert result.ok, result.for_model()
    assert result.data is not None
    assert result.data.price > 0


# ------------------------------------------------------------------ YFinanceProvider
# 用假的 yfinance 模块驱动，验证"fast_info 静默返回 None"时的两条路径。


class _FakeFastInfo:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def get(self, key: str) -> Any:
        return self._values.get(key)


class _FakeTicker:
    def __init__(self, fast: dict[str, Any] | Exception, frame: Any) -> None:
        self._fast = fast
        self._frame = frame

    @property
    def fast_info(self) -> Any:
        if isinstance(self._fast, Exception):
            raise self._fast
        return _FakeFastInfo(self._fast)

    def history(self, **_: Any) -> Any:
        if isinstance(self._frame, Exception):
            raise self._frame
        return self._frame


def _install_fake_yfinance(monkeypatch: pytest.MonkeyPatch, ticker_obj: _FakeTicker) -> None:
    import sys
    import types

    module = types.ModuleType("yfinance")
    module.Ticker = lambda _symbol: ticker_obj  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", module)


def test_yfinance_uses_fast_info_when_it_works(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_yfinance(
        monkeypatch,
        _FakeTicker({"last_price": 184.3, "currency": "USD"}, frame=None),
    )
    payload = market.YFinanceProvider().quote("NVDA")
    assert payload["price"] == pytest.approx(184.3)
    assert payload["currency"] == "USD"


def test_yfinance_falls_back_to_history_when_fast_info_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实世界最常见的一种失败：fast_info 平静地返回 None。

    库不抛异常，所以只判断"是不是 None"会让整条链路无声地失败。
    """
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {"Close": [180.0, 184.3]},
        index=pd.to_datetime(["2026-08-25", "2026-08-26"]),
    )
    _install_fake_yfinance(monkeypatch, _FakeTicker({"last_price": None}, frame=frame))

    payload = market.YFinanceProvider().quote("NVDA")
    assert payload["price"] == pytest.approx(184.3)
    assert payload["as_of"].startswith("2026-08-26"), "用行情自己的日期，不是 now()"


def test_yfinance_error_names_every_path_it_tried(monkeypatch: pytest.MonkeyPatch) -> None:
    """报错必须说清楚是限流还是代码写错——不然只能干瞪眼。"""
    _install_fake_yfinance(
        monkeypatch,
        _FakeTicker({"last_price": None}, frame=ConnectionError("429 Too Many Requests")),
    )
    with pytest.raises(RuntimeError) as excinfo:
        market.YFinanceProvider().quote("NVDA")

    message = str(excinfo.value)
    assert "fast_info" in message and "返回 None" in message
    assert "history" in message and "429" in message


def test_as_utc_attaches_a_timezone() -> None:
    """naive 的时间会被 Source.retrieved_at 直接拒掉。"""
    from datetime import datetime

    stamp = market._as_utc(datetime(2026, 8, 26, 12, 0))
    assert stamp.tzinfo is not None
