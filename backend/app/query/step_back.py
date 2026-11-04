from app.query.base import QueryTransform
from app.llm import generate


class StepBackTransform(QueryTransform):
    """
    Step-Back Prompting:
    Asks a higher-level, broader conceptual question to retrieve background concepts.
    """

    def transform(self, query: str) -> list[str]:
        prompt = f"""You are an expert in computer science and software development.
Your task is to step back and paraphrase the user's query into a more generic, high-level conceptual question.

Example:
Original: "How do I define path params in FastAPI with Enum?"
Step-Back: "How does FastAPI handle path parameters and validation?"

Original Question: {query}
Step-Back Question:"""

        step_back = generate(prompt).strip()
        return [query, step_back]