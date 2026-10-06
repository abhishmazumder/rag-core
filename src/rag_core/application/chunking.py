import re

_PARAGRAPH_BOUNDARY = re.compile(r"\n[ \t]*\n+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_BOUNDARY = re.compile(r"\s+")


def fixed_size_chunks(
    text: str,
    *,
    chunk_size_chars: int,
    overlap_chars: int = 0,
) -> list[str]:
    _validate_chunk_settings(chunk_size_chars, overlap_chars)
    if not text.strip():
        return []

    chunks: list[str] = []
    step = chunk_size_chars - overlap_chars
    previous_end = 0
    for start in range(0, len(text), step):
        end = min(start + chunk_size_chars, len(text))
        if end <= previous_end:
            continue
        chunk = text[start:end]
        chunks.append(chunk)
        previous_end = start + len(chunk)
    return chunks


def recursive_chunks(
    text: str,
    *,
    chunk_size_chars: int,
    overlap_chars: int = 0,
) -> list[str]:
    _validate_chunk_settings(chunk_size_chars, overlap_chars)
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        window_end = min(start + chunk_size_chars, len(text))
        end = _preferred_boundary(text, start, window_end)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def _validate_chunk_settings(chunk_size_chars: int, overlap_chars: int) -> None:
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be greater than 0.")
    if overlap_chars < 0 or overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be at least 0 and less than chunk_size_chars.")


def _preferred_boundary(text: str, start: int, window_end: int) -> int:
    if window_end == len(text):
        return window_end

    for boundary_pattern in (
        _PARAGRAPH_BOUNDARY,
        _SENTENCE_BOUNDARY,
        _WHITESPACE_BOUNDARY,
    ):
        boundaries = [
            match.end()
            for match in boundary_pattern.finditer(text, start, window_end)
            if match.end() > start
        ]
        if boundaries:
            return boundaries[-1]

    return window_end
