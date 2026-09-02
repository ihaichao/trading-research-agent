"""令牌桶限速。

SEC 明确规定 10 请求/秒，超了封 IP——不是限流，是封禁。
"""

from __future__ import annotations

import threading
import time

from tra.config import get_settings


class RateLimiter:
    """令牌桶：以固定速率产生令牌，取不到就阻塞等待。"""

    def __init__(self, rate_per_sec: float, burst: float | None = None) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec 必须大于 0")
        self.rate = rate_per_sec
        self.capacity = burst if burst is not None else rate_per_sec
        self._tokens = self.capacity
        # monotonic 而不是 time()：前者不受系统时钟调整影响，
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> float:
        """取令牌，不够就阻塞。返回实际等待的秒数（测试和监控用）。"""
        if tokens > self.capacity:
            raise ValueError(f"一次最多取 {self.capacity} 个令牌，请求了 {tokens}")

        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited

                shortfall = tokens - self._tokens
                sleep_for = shortfall / self.rate

            time.sleep(sleep_for)
            waited += sleep_for


_sec_limiter: RateLimiter | None = None
_sec_lock = threading.Lock()


def sec_limiter() -> RateLimiter:
    """进程级共享的 SEC 限速器。所有打 sec.gov 的代码都必须走它。"""
    global _sec_limiter
    with _sec_lock:
        if _sec_limiter is None:
            _sec_limiter = RateLimiter(float(get_settings().sec_rate_limit_per_sec))
        return _sec_limiter
