import re
import unicodedata


def clean_markdown(text: str) -> str:
    """
    Cleans markdown text before chunking:
    - Normalizes unicode characters.
    - Strips YAML frontmatter (--- ... ---).
    - Removes HTML comments and markdown navigation junk.
    - Collapses excessive empty lines into double newlines.
    """
    # 1. Normalize unicode (NFKC replaces weird spaces/accents cleanly)
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip YAML frontmatter at top of file
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)

    # 3. Strip HTML comments (<!-- ... -->)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 4. Remove FastAPI-specific macro tags like '{* ... *}' and '/// note'
    text = re.sub(r"\{\*.*?\*\}", "", text)
    text = re.sub(r"///.*?\n", "", text)

    # 5. Normalize whitespace (keep paragraph breaks, remove trailing spaces)
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()