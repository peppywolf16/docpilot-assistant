REFUSAL_PHRASE = "i don't have enough information in the provided documents"


def is_passage_relevant(passage: dict, gold_doc: str, gold_passage: str | None = None) -> bool:
    """Checks if a retrieved passage matches the gold document and optional snippet."""
    if not gold_doc:
        return False

    # Safely convert to string in case doc_name is stored as an int or None
    doc_name = str(passage.get("doc", "")).lower()
    gold_doc_str = str(gold_doc).lower()
    
    doc_matches = (doc_name == gold_doc_str) or (gold_doc_str in doc_name)
    if not doc_matches:
        return False

    # If gold_passage snippet is specified, verify it appears in the chunk text
    if gold_passage:
        chunk_text = str(passage.get("text", "")).lower()
        return str(gold_passage).lower() in chunk_text

    return True


def calculate_hit_at_k(passages: list[dict], gold_doc: str | None, gold_passage: str | None = None, k: int = 4) -> float:
    """Returns 1.0 if any of top-k passages matches gold doc/passage, else 0.0."""
    if not gold_doc:
        return 1.0 if len(passages) == 0 else 0.0

    for p in passages[:k]:
        if is_passage_relevant(p, gold_doc, gold_passage):
            return 1.0
    return 0.0


def calculate_mrr(passages: list[dict], gold_doc: str | None, gold_passage: str | None = None, k: int = 4) -> float:
    """Returns Reciprocal Rank (1/rank) of the FIRST relevant passage in top-k."""
    if not gold_doc:
        return 1.0 if len(passages) == 0 else 0.0

    for rank, p in enumerate(passages[:k], start=1):
        if is_passage_relevant(p, gold_doc, gold_passage):
            return 1.0 / rank
    return 0.0


def check_abstention(answer: str, is_unanswerable: bool) -> tuple[bool, str]:
    """
    Checks if answer refused appropriately:
    Catches both "don't have enough information" and "do not have enough information".
    """
    answer_lower = answer.lower()
    refused = (
        "don't have enough information" in answer_lower
        or "do not have enough information" in answer_lower
        or "cannot be found in the provided passages" in answer_lower
    )

    if is_unanswerable:
        if refused:
            return True, "Correct refusal"
        else:
            return False, "Hallucinated on unanswerable question"
    else:
        if refused:
            return False, "False refusal on answerable question"
        else:
            return True, "Correctly answered"