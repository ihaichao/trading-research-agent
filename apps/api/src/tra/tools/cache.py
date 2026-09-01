"""磁盘缓存：同样的请求不打第二次。

**只缓存上游原始数据，不缓存领域对象。**
被装饰的函数返回值必须是 JSON 可序列化的（dict / list / str / int / float / bool / None）。
"""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from tra.config import get_settings

P = ParamSpec("P")
R = TypeVar("R")


def cache_root() -> Path:
    """缓存根目录。每次调用都重新读配置，测试里才好替换。"""
    return Path(get_settings().cache_dir)


def cache_path(namespace: str, key: str) -> Path:
    return cache_root() / namespace / f"{key}.json"


def _make_key(func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """由函数名 + 参数算出稳定的缓存键。

    `default=str` 让不可 JSON 化的参数退化成字符串而不是报错；
    `sort_keys=True` 保证 dict 参数的键序不影响缓存键。
    """
    payload = json.dumps([func_name, args, kwargs], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def disk_cache(
    *,
    namespace: str,
    ttl: timedelta | None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """把函数结果缓存到磁盘。

    ttl=None 表示永不过期。

    用法::

        @disk_cache(namespace="sec", ttl=timedelta(days=1))
        def fetch_company_facts(cik: str) -> dict:
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)  # 保留原函数的名字和 docstring
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            path = cache_path(namespace, _make_key(func.__qualname__, args, kwargs))

            if path.exists():
                try:
                    envelope = json.loads(path.read_text(encoding="utf-8"))
                    stored_at = datetime.fromisoformat(envelope["stored_at"])
                    if ttl is None or datetime.now(UTC) - stored_at < ttl:
                        return envelope["value"]
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass  # 缓存文件损坏就当没有，重新取一次

            value = func(*args, **kwargs)

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"stored_at": datetime.now(UTC).isoformat(), "value": value},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return value

        return wrapper

    return decorator


def clear_cache(namespace: str | None = None) -> None:
    """清空某个命名空间（或全部）的缓存。调试时用。"""
    target = cache_root() / namespace if namespace else cache_root()
    if target.exists():
        shutil.rmtree(target)
