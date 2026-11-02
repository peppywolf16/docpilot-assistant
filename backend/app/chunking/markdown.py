import re
from app.chunking.base import Chunker
from app.chunking.cleaner import clean_markdown


class MarkdownChunker(Chunker):
    """
    Structure-aware chunker for technical Markdown:
    - Tracks heading hierarchy (#, ##, ###).
    - Prepends [Section: H1 > H2] to every chunk for context.
    - Never slices code blocks (``` ... ```) in half.
    """

    def split(self, text: str, metadata: dict | None = None) -> list[dict]:
        cleaned_text = clean_markdown(text)
        if not cleaned_text:
            return []

        lines = cleaned_text.split("\n")
        sections = []
        heading_stack = []

        current_section_lines = []
        in_code_block = False

        for line in lines:
            # Track if we are inside a code block
            if line.strip().startswith("```"):
                in_code_block = not in_code_block

            # Check for Markdown headers (only outside code blocks!)
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line) if not in_code_block else None

            if header_match:
                # Flush previous section lines
                if current_section_lines:
                    heading_path = " > ".join(heading_stack) if heading_stack else "Introduction"
                    sections.append((heading_path, "\n".join(current_section_lines).strip()))
                    current_section_lines = []

                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                # Clean FastAPI heading anchor tags like {#path-params}
                title = re.sub(r"\{#.*?\}", "", title).strip()

                # Adjust heading stack according to header depth
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(title)
            else:
                current_section_lines.append(line)

        # Flush final section
        if current_section_lines:
            heading_path = " > ".join(heading_stack) if heading_stack else "General"
            sections.append((heading_path, "\n".join(current_section_lines).strip()))

        # Now subdivide oversized sections while protecting code blocks
        chunks = []
        position = 1

        for heading_path, section_text in sections:
            if not section_text:
                continue

            section_chunks = self._chunk_section(section_text)
            for s_chunk in section_chunks:
                # Prepend the heading path as an anchor
                context_header = f"[Section: {heading_path}]"
                full_text = f"{context_header}\n\n{s_chunk}".strip()

                chunks.append({
                    "text": full_text,
                    "heading_path": heading_path,
                    "char_count": len(full_text),
                    "position": position,
                })
                position += 1

        return chunks

    def _chunk_section(self, text: str) -> list[str]:
        """Splits an individual section by paragraphs, keeping code blocks intact."""
        if len(text) <= self.chunk_size:
            return [text]

        # Break text into atomic blocks (paragraphs and code blocks)
        blocks = []
        curr_block = []
        in_code = False

        for line in text.split("\n"):
            if line.strip().startswith("```"):
                in_code = not in_code
                curr_block.append(line)
                if not in_code:  # finished code block
                    blocks.append("\n".join(curr_block))
                    curr_block = []
            elif not in_code and line.strip() == "":
                if curr_block:
                    blocks.append("\n".join(curr_block))
                    curr_block = []
            else:
                curr_block.append(line)

        if curr_block:
            blocks.append("\n".join(curr_block))

        # Combine atomic blocks up to chunk_size
        results = []
        curr_chunk = ""

        for b in blocks:
            if not curr_chunk:
                curr_chunk = b
            elif len(curr_chunk) + len(b) + 2 <= self.chunk_size:
                curr_chunk += "\n\n" + b
            else:
                results.append(curr_chunk)
                curr_chunk = b

        if curr_chunk:
            results.append(curr_chunk)

        return results