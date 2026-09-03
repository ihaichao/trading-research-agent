"""行情数据：报价与历史价格。

1. 数据源抽象成 provider 接口，按顺序尝试，一个挂了换下一个
2. 所有结果落盘缓存，挂掉时至少还有上次的数据
3. 全挂时返回一条说明白的 ToolResult，让模型如实报告缺口，而不是编一个价格

"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from pydantic import BaseModel

from tra.report.schema import MetricPoint, MetricSeries, Source, SourceKind, make_source_id
from tra.tools.base import ToolResult
from tra.tools.cache import disk_cache

QUOTE_TTL = timedelta(minutes=15)
HISTORY_TTL = timedelta(days=1)


def _as_utc(value: Any) -> datetime:
    """把 pandas Timestamp / datetime 归一成带时区的 datetime。

    Source.retrieved_at 要求 tz-aware，naive 的时间会被 schema 直接拒掉。
    """
    stamp = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    if not isinstance(stamp, datetime):
        return datetime.now(UTC)
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


class Quote(BaseModel):
    ticker: str
    price: float
    currency: str = "USD"
    as_of: datetime
    provider: str


class PriceProvider(Protocol):
    """行情源接口。返回的必须是 JSON 可序列化的裸数据，好让缓存层直接落盘。"""

    name: str

    def quote(self, ticker: str) -> dict[str, Any]: ...

    def history(self, ticker: str, *, period: str, interval: str) -> list[dict[str, Any]]: ...


class YFinanceProvider:
    """Yahoo Finance（非官方）。免费、无需 key，但不稳定。

    一个必须知道的坑：**`fast_info` 拿不到数据时静默返回 None，不抛异常。**
    网络失败、被限流、cookie/crumb 取不到——全都表现为一个平静的 None。
    所以这里走两条路径，并把每条路径的失败原因都记下来一起往上抛：
    "取不到价格"的报错必须说清是限流还是代码写错了，否则你只能干瞪眼。
    """

    name = "yfinance"

    def quote(self, ticker: str) -> dict[str, Any]:
        import yfinance as yf

        handle = yf.Ticker(ticker)
        attempts: list[str] = []
        currency = "USD"

        # 路径 1：fast_info，最便宜，但会静默返回 None
        try:
            info = handle.fast_info
            currency = str(info.get("currency") or "USD")
            price = info.get("last_price") or info.get("previous_close")
            if price is not None:
                return {
                    "price": float(price),
                    "currency": currency,
                    "as_of": datetime.now(UTC).isoformat(),
                }
            attempts.append("fast_info: 返回 None（Yahoo 限流或 cookie 取不到时的典型表现）")
        except Exception as exc:
            attempts.append(f"fast_info: {type(exc).__name__}: {exc}")

        # 路径 2：history，走行情图接口，失败时会真的抛异常
        try:
            frame = handle.history(period="5d")
            if frame is not None and not frame.empty:
                return {
                    "price": float(frame["Close"].iloc[-1]),
                    "currency": currency,
                    "as_of": _as_utc(frame.index[-1]).isoformat(),
                }
            attempts.append("history: 返回空表")
        except Exception as exc:
            attempts.append(f"history: {type(exc).__name__}: {exc}")

        raise RuntimeError(f"拿不到 {ticker!r} 的价格。尝试过 —— " + " | ".join(attempts))

    def history(self, ticker: str, *, period: str, interval: str) -> list[dict[str, Any]]:
        import yfinance as yf

        frame = yf.Ticker(ticker).history(period=period, interval=interval)
        if frame is None or frame.empty:
            raise ValueError(f"no price history returned for {ticker!r}")
        return [
            {"date": index.date().isoformat(), "close": float(row["Close"])}
            for index, row in frame.iterrows()
        ]


# 按顺序尝试。加备用源（Tiingo / Alpha Vantage）就往这个列表里追加一个类。
PROVIDERS: list[PriceProvider] = [YFinanceProvider()]


def yahoo_url(ticker: str) -> str:
    return f"https://finance.yahoo.com/quote/{ticker.upper()}"


@disk_cache(namespace="market_quote", ttl=QUOTE_TTL)
def _cached_quote(provider_name: str, ticker: str) -> dict[str, Any]:
    provider = next(p for p in PROVIDERS if p.name == provider_name)
    return provider.quote(ticker)


@disk_cache(namespace="market_history", ttl=HISTORY_TTL)
def _cached_history(
    provider_name: str, ticker: str, period: str, interval: str
) -> list[dict[str, Any]]:
    provider = next(p for p in PROVIDERS if p.name == provider_name)
    return provider.history(ticker, period=period, interval=interval)


def _try_providers(call: Any, *args: Any) -> tuple[str, Any] | list[str]:
    """按顺序试每个 provider，返回 (provider_name, 结果) 或全部失败的原因列表。"""
    failures: list[str] = []
    for provider in PROVIDERS:
        try:
            return provider.name, call(provider.name, *args)
        except Exception as exc:
            failures.append(f"{provider.name}: {type(exc).__name__}: {exc}")
    return failures


def get_quote(ticker: str) -> ToolResult[Quote]:
    """最新收盘/最近价格。延迟数据，不是实时行情。"""
    symbol = ticker.strip().upper()
    outcome = _try_providers(_cached_quote, symbol)

    if isinstance(outcome, list):
        return ToolResult.failure(
            f"all price providers failed for {symbol!r}: {'; '.join(outcome)}",
            hint=(
                "do not estimate a price. Report that the quote is unavailable, "
                "or answer using SEC fundamentals only."
            ),
        )

    provider_name, payload = outcome
    as_of = datetime.fromisoformat(payload["as_of"])
    quote = Quote(
        ticker=symbol,
        price=float(payload["price"]),
        currency=str(payload.get("currency", "USD")),
        as_of=as_of,
        provider=provider_name,
    )

    url = yahoo_url(symbol)
    locator = f"{provider_name} quote {as_of.date().isoformat()}"
    source = Source(
        id=make_source_id(url, locator),
        kind=SourceKind.MARKET_DATA,
        url=url,
        title=f"{symbol} quote",
        locator=locator,
        snippet=f"{symbol} {quote.price:.2f} {quote.currency} (delayed, via {provider_name})",
        retrieved_at=as_of,
    )
    return ToolResult.success(quote, [source])


def get_price_history(
    ticker: str,
    *,
    period: str = "2y",
    interval: str = "1mo",
) -> ToolResult[MetricSeries]:
    """历史收盘价，直接吐成报告契约里的 MetricSeries。

    period: 1mo/3mo/6mo/1y/2y/5y/max ； interval: 1d/1wk/1mo
    """
    symbol = ticker.strip().upper()
    outcome = _try_providers(_cached_history, symbol, period, interval)

    if isinstance(outcome, list):
        return ToolResult.failure(
            f"all price providers failed for {symbol!r}: {'; '.join(outcome)}",
            hint="report the gap; do not reconstruct prices from memory.",
        )

    provider_name, rows = outcome
    if not rows:
        return ToolResult.failure(
            f"empty price history for {symbol!r} (period={period}, interval={interval})",
            hint="try a longer period, or check the ticker is still listed.",
        )

    url = yahoo_url(symbol)
    locator = f"{provider_name} close {period}/{interval}"
    source = Source(
        id=make_source_id(url, locator),
        kind=SourceKind.MARKET_DATA,
        url=url,
        title=f"{symbol} price history ({period}, {interval})",
        locator=locator,
        snippet=f"{len(rows)} closes from {rows[0]['date']} to {rows[-1]['date']}",
        retrieved_at=datetime.now(UTC),
    )
    series = MetricSeries(
        key="close_price",
        label=f"{symbol} close",
        unit="USD",
        points=[MetricPoint(period=r["date"], value=float(r["close"])) for r in rows],
        source_ids=[source.id],
    )
    return ToolResult.success(series, [source])
