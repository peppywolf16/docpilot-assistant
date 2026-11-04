import json
import re
from pydantic import BaseModel, Field
from app.llm import generate


class RouteClassification(BaseModel):
    route: str = Field(
        description="Must be one of: 'direct_chat' (greetings/general chat), 'search' (single technical question), or 'decompose' (multi-part/comparison question)"
    )
    reason: str = Field(description="Short rationale for the routing decision")


def route_query(query: str) -> str:
    """
    Classifies the user query into one of three routes:
    - 'direct_chat': Casual greetings, meta-questions (e.g. "Hi", "Who created you?")
    - 'search': Standard technical question requiring documentation search
    - 'decompose': Complex question with multiple parts or comparing multiple concepts
    """
    prompt = f"""You are an intent router for a technical documentation system.
Classify the user's input into exactly one of three categories:

1. "direct_chat": Greetings, pleasantries, or general questions about what you can do (e.g., "Hello", "Thanks", "Who are you?").
2. "search": A single technical question about documentation, APIs, or code.
3. "decompose": A complex multi-part question that asks about multiple distinct topics, steps, or comparisons.

Respond ONLY with a valid JSON object matching this schema:
{{
  "route": "direct_chat" | "search" | "decompose",
  "reason": "short explanation"
}}

User Input: {query}
JSON Output:"""

    raw_response = generate(prompt).strip()

    cleaned = re.sub(r"^```json\s*", "", raw_response)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        validated = RouteClassification(**data)
        if validated.route in ["direct_chat", "search", "decompose"]:
            return validated.route
        return "search"
    except Exception:
        return "search"