"""把 SEC 的 XBRL 原始数据下载到本地，供离线探索。

    uv run python examples/sec_fetch.py NVDA

写出两个文件到 .cache/sec/ ：
  tickers.json                  —— 全市场 ticker → CIK 映射表
  companyfacts_NVDA.json        —— 这家公司的全部 XBRL 财务数据
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx

from tra.config import get_settings

# SEC 官方端点。CIK 必须补零到整 10 位，例如 NVDA 的 1045810 -> "CIK0001045810"。
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

TIMEOUT = httpx.Timeout(30.0)


def cache_dir() -> Path:
    d = Path(get_settings().cache_dir) / "sec"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_agent() -> str:
    ua = get_settings().sec_user_agent
    return ua


def fetch_json(client: httpx.Client, url: str, dest: Path, *, force: bool = False) -> Any:
    """下载 JSON 到 dest；已存在就直接读本地，除非 force=True。"""
    if dest.exists() and not force:
        print(f"  ✓ 命中本地缓存 {dest}（加 --force 可强制重下）")
        return json.loads(dest.read_text(encoding="utf-8"))

    print(f"  ↓ GET {url}")
    resp = client.get(url)
    resp.raise_for_status()  # 4xx/5xx 直接抛，别让坏数据往下流
    data = resp.json()

    dest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ 已写入 {dest}（{dest.stat().st_size / 1024:.0f} KB）")
    return data


def cik_for(ticker: str, tickers: dict[str, Any]) -> str:
    """从映射表里找出 CIK，并补零到 10 位。

    tickers.json 的结构是 {"0": {"cik_str": 1045810, "ticker": "NVDA", ...}, "1": {...}}
    —— 外层的 key 是无意义的序号，真正要找的东西在 value 里。
    """
    target = ticker.strip().upper()
    for entry in tickers.values():
        if entry["ticker"].upper() == target:
            return str(entry["cik_str"]).zfill(10)  # zfill = 左侧补零
    raise SystemExit(f"在 SEC 的映射表里找不到 ticker {target!r}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    force = "--force" in sys.argv
    ticker = (args[0] if args else "NVDA").upper()

    out = cache_dir()
    with httpx.Client(headers={"User-Agent": user_agent()}, timeout=TIMEOUT) as client:
        print("1) ticker → CIK 映射表")
        tickers = fetch_json(client, TICKERS_URL, out / "tickers.json", force=force)

        cik = cik_for(ticker, tickers)
        print(f"   {ticker} 的 CIK = {cik}")

        print(f"2) {ticker} 的 XBRL 财务数据")
        facts_path = out / f"companyfacts_{ticker}.json"
        facts = fetch_json(client, COMPANY_FACTS_URL.format(cik=cik), facts_path, force=force)

    print(f"   顶层 key: {list(facts.keys())}")
    for taxonomy, concepts in facts.get("facts", {}).items():
        print(f"   facts.{taxonomy}: {len(concepts)} 个概念")


if __name__ == "__main__":
    main()
