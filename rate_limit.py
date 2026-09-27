"""Redis-based rate limiting for FinGuard Agent.

Fallback về in-memory nếu Redis không khả dụng.
"""

from __future__ import annotations

import os
import time

try:
    import redis

    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

WINDOW_SEC = 60  # Cửa sổ 60 giây
MAX_REQUESTS = 10  # Tối đa 10 requests / window
KEY_PREFIX = "finaigent:rl:"


# ============================================================
# SINGLETON REDIS CLIENT
# ============================================================
_redis_client: redis.Redis | None = None
_redis_failed: bool = False

# In-memory fallback (per-process)
_memory_store: dict[str, list[float]] = {}


def _get_redis_client():
    """Lazy-init Redis client. Trả None nếu không kết nối được."""
    global _redis_client, _redis_failed

    if _redis_client is not None:
        return _redis_client
    if _redis_failed or not _REDIS_AVAILABLE:
        return None

    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        _redis_client = client
        return _redis_client
    except Exception:
        _redis_failed = True
        return None


# ============================================================
# CORE LOGIC
# ============================================================
def _check_redis(client, user_id: str) -> tuple[bool, int]:
    """Kiểm tra rate limit bằng Redis sorted set.

    Trả về (allowed, remaining).
    """
    key = f"{KEY_PREFIX}{user_id}"
    now = time.time()
    window_start = now - WINDOW_SEC

    pipe = client.pipeline()
    # Xóa các entry ngoài window
    pipe.zremrangebyscore(key, 0, window_start)
    # Đếm số request trong window hiện tại
    pipe.zcard(key)
    # Thêm request hiện tại
    pipe.zadd(key, {str(now): now})
    # Set expiry cho key
    pipe.expire(key, WINDOW_SEC + 1)
    results = pipe.execute()

    count = results[1]
    allowed = count < MAX_REQUESTS
    remaining = max(0, MAX_REQUESTS - count - 1)

    return allowed, remaining


def _check_memory(user_id: str) -> tuple[bool, int]:
    """Fallback in-memory rate limit."""
    now = time.time()
    window_start = now - WINDOW_SEC

    times = _memory_store.get(user_id, [])
    times = [t for t in times if t > window_start]

    allowed = len(times) < MAX_REQUESTS
    if allowed:
        times.append(now)
    _memory_store[user_id] = times

    remaining = max(0, MAX_REQUESTS - len(times))
    return allowed, remaining


# ============================================================
# PUBLIC API
# ============================================================
def check_rate_limit(user_id: str = "default") -> tuple[bool, int]:
    """Kiểm tra rate limit cho user.

    Trả về (allowed, remaining_requests).
    """
    client = _get_redis_client()
    if client is not None:
        try:
            return _check_redis(client, user_id)
        except Exception:
            # Nếu Redis lỗi giữa chừng, fallback memory
            return _check_memory(user_id)
    return _check_memory(user_id)


def get_backend_name() -> str:
    """Trả về backend đang dùng: 'redis' hoặc 'memory'."""
    return "redis" if _get_redis_client() is not None else "memory"


def reset_user(user_id: str = "default") -> None:
    """Xóa rate limit cho user (dùng cho testing)."""
    client = _get_redis_client()
    if client is not None:
        try:
            client.delete(f"{KEY_PREFIX}{user_id}")
            return
        except Exception:
            pass
    _memory_store.pop(user_id, None)


def get_redis_status() -> dict:
    """Trả về thông tin trạng thái Redis cho UI."""
    client = _get_redis_client()
    if client is None:
        return {
            "backend": "memory",
            "host": "-",
            "window_sec": WINDOW_SEC,
            "max_requests": MAX_REQUESTS,
        }
    return {
        "backend": "redis",
        "host": f"{REDIS_HOST}:{REDIS_PORT}",
        "window_sec": WINDOW_SEC,
        "max_requests": MAX_REQUESTS,
    }
