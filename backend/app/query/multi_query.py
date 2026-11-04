from app.query.base import QueryTransform
from app.llm import generate


class MultiQueryTransform(QueryTransform):
    """Generates 3 distinct rephrasings of the input query to maximize search coverage."""

    def transform(self, query: str) -> list[str]:
        prompt = f"""You are a technical documentation search assistant.
Your task is to generate 3 different versions of the user's query to retrieve relevant documentation.
Provide distinct perspectives focusing on alternative keywords, function names, and technical phrasing.

User Question: {query}

Provide exactly 3 rephrased queries, one per line. Do not number them:"""

        response = generate(prompt)
        rephrased = [line.strip("- *0123456789. ") for line in response.strip().split("\n") if line.strip()]
        
        # Always include the original query first
        return [query] + rephrased[:3]