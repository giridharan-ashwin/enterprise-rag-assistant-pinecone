import re


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    Split Markdown/text into semantically meaningful chunks.

    Each chunk contains:
    - section: heading associated with the content
    - text: chunk content
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    normalized = text.replace("\r\n", "\n").strip()

    if not normalized:
        return []

    sections: list[dict] = []

    current_section = "Document"
    current_lines: list[str] = []

    for line in normalized.split("\n"):
        heading_match = HEADING_PATTERN.match(line)

        if heading_match:
            # Save content collected before this heading.
            if current_lines:
                sections.append(
                    {
                        "section": current_section,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
                current_lines = []

            current_section = heading_match.group(2).strip()
            continue

        current_lines.append(line)

    # Save the final section.
    if current_lines:
        sections.append(
            {
                "section": current_section,
                "text": "\n".join(current_lines).strip(),
            }
        )

    final_chunks: list[dict] = []

    for section in sections:
        section_text = section["text"].strip()

        if not section_text:
            continue

        section_chunks = _split_large_section(
            section=section_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in section_chunks:
            final_chunks.append(
                {
                    "section": section["section"],
                    "text": chunk,
                }
            )

    return final_chunks


def _split_large_section(
    section: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Preserve a complete section when possible.
    Split oversized sections with overlap.
    """

    normalized = " ".join(section.split())

    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []

    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))

        chunk = normalized[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(normalized):
            break

        start = end - overlap

    return chunks