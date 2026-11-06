import json
import re
from app.generation.schemas import GroundedResponse
from app.llm import generate
from app.generation.prompts import PROMPT_V2_STRUCTURED


def verify_citations(grounded_resp: GroundedResponse, passages: list[dict]) -> tuple[GroundedResponse, float]:
    """
    Verifies every citation:
    1. Checks if the passage_number exists in retrieved passages.
    2. Checks if exact_quote actually appears in that passage's text.
    
    Returns:
    - verified_response: GroundedResponse with hallucinated quotes stripped out.
    - validity_rate: Percentage of valid citations (1.0 = 100% trustworthy).
    """
    if not grounded_resp.citations:
        return grounded_resp, 1.0

    passage_map = {p["number"]: p["text"] for p in passages}
    valid_citations = []

    for c in grounded_resp.citations:
        p_text = passage_map.get(c.passage_number)
        if not p_text:
            continue  # Passage number does not exist

        # Check if the exact quote (or normalized version) is in the passage
        quote_clean = c.exact_quote.strip().lower()
        if quote_clean and quote_clean in p_text.lower():
            valid_citations.append(c)

    validity_rate = len(valid_citations) / len(grounded_resp.citations) if grounded_resp.citations else 1.0
    grounded_resp.citations = valid_citations
    return grounded_resp, round(validity_rate, 3)


def generate_verified_answer(
    question: str, 
    passages: list[dict],
    rerank_threshold: float = 0.0
) -> dict:
    """
    Day 12 Verified Answer Generation:
    1. Threshold Check: If top passage score < threshold, refuse immediately (0 LLM call!).
    2. Generate structured JSON with quotes.
    3. Verify quotes in code.
    """
    # 1. Fast Refusal if search confidence is too low
    if not passages or (passages[0].get("score", 0.0) < rerank_threshold):
        return {
            "answerable": False,
            "answer": f"I searched the documentation for '{question}', but found no relevant sections matching your query.",
            "citations": [],
            "citation_validity": 1.0,
            "sources": passages,
            "fast_refusal": True
        }

    # 2. Build context
    context = "\n\n".join([f"[{p['number']}] (Source: {p['doc']})\n{p['text']}" for p in passages])
    prompt = PROMPT_V2_STRUCTURED.format(context=context, question=question)

    raw_response = generate(prompt).strip()
    cleaned = re.sub(r"^```json\s*", "", raw_response)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        resp_obj = GroundedResponse(**data)
    except Exception:
        # Fallback if JSON parsing fails
        return {
            "answerable": False,
            "answer": "I don't have enough information in the provided documents.",
            "citations": [],
            "citation_validity": 1.0,
            "sources": passages,
            "fast_refusal": False
        }

    # 3. Verify citations in code
    verified_obj, validity_rate = verify_citations(resp_obj, passages)

    return {
        "answerable": verified_obj.answerable,
        "answer": verified_obj.answer,
        "citations": [c.model_dump() for c in verified_obj.citations],
        "citation_validity": validity_rate,
        "sources": passages,
        "fast_refusal": False
    }