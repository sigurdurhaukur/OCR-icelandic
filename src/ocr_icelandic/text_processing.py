"""Text processing utilities for synthetic OCR generation."""


def split_long_text(text: str, max_length: int) -> list[str]:
    """
    Split text into chunks at sentence boundaries.

    Args:
        text: The text to split
        max_length: Maximum length for each chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    sentences = (
        text.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|")
    )

    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def normalize_range(
    min_value: int, max_value: int, minimum: int = 1
) -> tuple[int, int]:
    """
    Ensure min/max values form a valid range.

    Args:
        min_value: Minimum value
        max_value: Maximum value
        minimum: Minimum allowed value

    Returns:
        Normalized (min_value, max_value) tuple
    """
    min_value = max(minimum, min_value)
    max_value = max(min_value, max_value)
    return min_value, max_value
