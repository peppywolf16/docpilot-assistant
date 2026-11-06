import re

# Common patterns for credentials, tokens, and PII
PATTERNS = {
    "GOOGLE_API_KEY": (r"AIza[0-9A-Za-z-_]{30,40}", "[REDACTED_GOOGLE_API_KEY]"),
    "OPENAI_API_KEY": (r"sk-[a-zA-Z0-9]{32,}", "[REDACTED_OPENAI_API_KEY]"),
    "GENERIC_SECRET": (r"(?i)(password|secret|api_key|token)\s*[:=]\s*['\"][^\s'\"]+['\"]", r"\1=[REDACTED_SECRET]"),
    "BEARER_TOKEN": (r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}", "Bearer [REDACTED_TOKEN]"),
    "EMAIL": (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]"),
    "IPV4": (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[REDACTED_IP]"),
}


def mask_pii(text: str) -> str:
    """
    Masks sensitive personal identifiable information and leaked API keys.
    Run this at ingestion time before chunks are embedded into Qdrant/BM25.
    """
    masked_text = text
    for pattern_name, (regex, replacement) in PATTERNS.items():
        masked_text = re.sub(regex, replacement, masked_text)
    return masked_text


def detect_pii(text: str) -> dict[str, list[str]]:
    """Detects and returns all leaked PII items found in text without modifying it."""
    found = {}
    for pattern_name, (regex, _) in PATTERNS.items():
        matches = re.findall(regex, text)
        if matches:
            found[pattern_name] = matches
    return found