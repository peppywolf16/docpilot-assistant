import json
import re
from pydantic import BaseModel, Field
from app.llm import generate


class QueryFilterResult(BaseModel):
    doc_filter: str | None = Field(
        default=None, 
        description="The exact filename from available documents if the user question explicitly mentions or targets it, otherwise null"
    )
    clean_query: str = Field(
        description="The search question with explicit file references stripped out for clean semantic search"
    )


def extract_query_filters(query: str, available_docs: list[str]) -> tuple[str | None, str]:
    """
    Extracts metadata filters (like document name) from natural language questions.
    Returns: (doc_filter, clean_query)
    """
    if not available_docs:
        return None, query

    docs_list_str = ", ".join([f"'{d}'" for d in available_docs])

    prompt = f"""You are a query filter extractor for a documentation search engine.
Available indexed documents: [{docs_list_str}]

Analyze the user's question below. If the user explicitly mentions or clearly targets one of the available documents, extract that exact filename as "doc_filter" and provide a "clean_query" with the document name removed.
If no specific document is mentioned, set "doc_filter": null and "clean_query" to the original question.

Respond ONLY with a valid JSON object matching this schema:
{{
  "doc_filter": "filename.md" or null,
  "clean_query": "cleaned question"
}}

Example:
Question: "In extra-models.md, how do I hide passwords?"
JSON Output:
{{
  "doc_filter": "extra-models.md",
  "clean_query": "How do I hide passwords?"
}}

Question: {query}
JSON Output:"""

    raw_response = generate(prompt).strip()

    cleaned = re.sub(r"^```json\s*", "", raw_response)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        validated = QueryFilterResult(**data)
        
        # Verify extracted doc is actually in our available documents
        if validated.doc_filter and validated.doc_filter in available_docs:
            return validated.doc_filter, validated.clean_query
        return None, query
    except Exception:
        return None, query