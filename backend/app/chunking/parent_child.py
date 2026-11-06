import uuid
from app.chunking.cleaner import clean_markdown
from app.chunking.markdown import MarkdownChunker


def split_parent_child(
    text: str,
    doc_name: str,
    parent_size: int = 1200,
    child_size: int = 250,
    child_overlap: int = 50,
) -> tuple[dict[str, str], list[dict]]:
    """
    Splits text into:
    1. Parents: larger conceptual blocks (~1200 chars).
    2. Children: small, focused chunks (~250 chars) linked to their parent via parent_id.
    
    Returns:
    - parents_lookup: dict of {parent_id: parent_text}
    - child_chunks: list of chunk dicts for Qdrant embedding
    """
    cleaned = clean_markdown(text)
    if not cleaned:
        return {}, []

    # 1. Create Parent sections using MarkdownChunker with parent_size
    parent_chunker = MarkdownChunker(chunk_size=parent_size, overlap=100)
    parent_dicts = parent_chunker.split(cleaned)

    parents_lookup = {}
    child_chunks = []

    for p_idx, p_data in enumerate(parent_dicts, start=1):
        parent_id = f"{doc_name}::parent_{p_idx}"
        parent_text = p_data["text"]
        heading_path = p_data.get("heading_path", "General")

        parents_lookup[parent_id] = parent_text

        # 2. Slice the parent into small child chunks for high-precision search
        start = 0
        c_idx = 1
        while start < len(parent_text):
            end = start + child_size
            child_text = parent_text[start:end].strip()

            if child_text:
                # Child inherits text directly from parent (which already has the heading path)
                child_chunks.append({
                    "doc_name": doc_name,
                    "parent_id": parent_id,
                    "child_id": f"{parent_id}::c_{c_idx}",
                    "text": child_text,
                    "heading_path": heading_path,
                })
                c_idx += 1

            if end >= len(parent_text):
                break
            start += child_size - child_overlap

    return parents_lookup, child_chunks



