import json
import re
from pydantic import BaseModel, Field
from app.llm import generate


class DecomposedQueries(BaseModel):
    is_complex: bool = Field(description="True if the question contains multiple distinct sub-questions")
    sub_questions: list[str] = Field(description="List of 2 to 3 atomic, independent sub-questions")


def decompose_query(query: str) -> list[str]:
    """
    Analyzes an input question. If it asks about multiple concepts or multi-step tasks,
    decomposes it into 2-3 focused sub-questions. Otherwise returns [query].
    """
    prompt = f"""You are a technical query planner.
Analyze the user question below. If it asks about multiple concepts, compares different features, or contains multiple parts, break it down into 2 or 3 standalone, focused sub-questions.
If it is a single focused question, set "is_complex": false and return only the original question.

Example 1:
Question: "How do path parameters differ from query parameters, and how do you declare request body?"
JSON Output:
{{
  "is_complex": true,
  "sub_questions": [
    "What is the difference between path parameters and query parameters in FastAPI?",
    "How do you declare a request body using Pydantic in FastAPI?"
  ]
}}

Example 2:
Question: "What is the command to run the development server?"
JSON Output:
{{
  "is_complex": false,
  "sub_questions": ["What is the command to run the development server?"]
}}

Question: {query}
JSON Output:"""

    raw_response = generate(prompt).strip()

    cleaned = re.sub(r"^```json\s*", "", raw_response)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        validated = DecomposedQueries(**data)
        if validated.is_complex and len(validated.sub_questions) > 1:
            return validated.sub_questions
        return [query]
    except Exception:
        return [query]