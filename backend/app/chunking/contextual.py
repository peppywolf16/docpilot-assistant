from app.llm import generate


def generate_contextual_prefix(doc_title: str, doc_summary: str, chunk_text: str) -> str:
    """
    Anthropic Contextual Retrieval:
    Generates a 1-2 sentence succinct context anchor to situate the chunk within the document.
    """
    prompt = f"""You are a technical documentation indexing assistant.
<document_title>
{doc_title}
</document_title>

<document_overview>
{doc_summary[:500]}
</document_overview>

<chunk>
{chunk_text}
</chunk>

Please provide a 1-sentence succinct context to situate this chunk within the overall document to improve search retrieval. Do not mention 'this chunk'. Answer only with the succinct context sentence:"""

    prefix = generate(prompt).strip()
    # Strip any wrapping quotes or markdown tags
    prefix = prefix.strip('"`\'')
    return prefix