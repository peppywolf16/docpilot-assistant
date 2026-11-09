import time
from diskcache import Cache
from app.config import INDEX_DIR

BUDGET_CACHE = Cache(str(INDEX_DIR / "tenant_budgets"))
DEFAULT_DAILY_TOKEN_LIMIT = 50_000  # 50k tokens/day per tenant


def _get_day_key(tenant: str) -> str:
    day_str = time.strftime("%Y-%m-%d")
    return f"budget:{tenant}:{day_str}"


def check_token_budget(tenant: str, requested_tokens: int = 100) -> tuple[bool, int, int]:
    """
    Checks if tenant has remaining daily token budget.
    Returns: (is_allowed, current_usage, limit)
    """
    key = _get_day_key(tenant)
    current_usage = BUDGET_CACHE.get(key, 0)
    limit = DEFAULT_DAILY_TOKEN_LIMIT

    if current_usage + requested_tokens > limit:
        return False, current_usage, limit

    return True, current_usage, limit


def record_token_usage(tenant: str, tokens_used: int):
    """Records consumed tokens against the tenant's daily quota."""
    key = _get_day_key(tenant)
    current = BUDGET_CACHE.get(key, 0)
    BUDGET_CACHE.set(key, current + tokens_used, expire=86400 * 2)