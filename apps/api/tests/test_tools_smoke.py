"""对真实外部接口的冒烟测试。默认不跑（CI 里用 -m 'not network' 排除）。

本地手动验证数据源是否还活着：
    uv run pytest -m network -q
"""

from __future__ import annotations

import pytest


@pytest.mark.network
def test_get_quote_returns_a_price() -> None:
    from examples.react_from_scratch import get_quote

    out = get_quote("NVDA")
    assert "NVDA" in out
    assert "ERROR" not in out
