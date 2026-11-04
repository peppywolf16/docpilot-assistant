from app.query.base import QueryTransform
from app.llm import generate


class HyDETransform(QueryTransform):
    """
    Hypothetical Document Embeddings (HyDE):
    Generates a hypothetical documentation passage that would answer the question,
    and returns it so it can be embedded in document space.
    """

    def transform(self, query: str) -> list[str]:
        prompt = f"""You are an expert technical documentation writer.
Write a clear, concise documentation paragraph (with code snippet if appropriate) that directly answers the question below.
Do not say 'Sure' or 'Here is...'. Write only the documentation passage itself.

Question: {query}
Documentation Passage:"""

        hypothetical_doc = generate(prompt).strip()
        # Returns the hypothetical document passage for embedding
        return [hypothetical_doc if hypothetical_doc else query]