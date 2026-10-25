import re

from app import config

# Sentence end (. ! ? … followed by whitespace) or a blank line between paragraphs
_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n\s*\n")


def _sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) character spans of sentences, without surrounding whitespace."""
    spans, start = [], 0
    for end, next_start in [(m.start(), m.end()) for m in _BOUNDARY.finditer(text)] + [(len(text), len(text))]:
        piece = text[start:end]
        if piece.strip():
            spans.append((start + len(piece) - len(piece.lstrip()), end - len(piece) + len(piece.rstrip())))
        start = next_start
    return spans


# ponytail: the regex splitter breaks on abbreviations like "Dr. Smith"; switch to nltk/pysbd if that matters
def chunk_spans(text: str, size: int = config.CHUNK_SIZE) -> list[tuple[int, int]]:
    """Packs sentences into chunks of up to `size` characters; neighbouring chunks overlap by one sentence.
    Each chunk is text[start:end], an exact slice of the original (so it can be highlighted in the source)."""
    chunks: list[tuple[int, int]] = []
    current: list[tuple[int, int]] = []
    for sentence in _sentences(text):
        if current and sentence[1] - current[0][0] > size:
            chunks.append((current[0][0], current[-1][1]))
            last = current[-1]
            current = [last] if sentence[1] - last[0] <= size else []
        current.append(sentence)
    if current:
        chunks.append((current[0][0], current[-1][1]))
    return chunks


if __name__ == "__main__":
    assert chunk_spans("") == []
    assert chunk_spans("  One sentence.  ") == [(2, 15)]
    text = "One two six. Four five 66! Seven eight 9?\n\nTen elevens"
    assert [text[s:e] for s, e in chunk_spans(text, size=35)] == [
        "One two six. Four five 66!",
        "Four five 66! Seven eight 9?",
        "Seven eight 9?\n\nTen elevens",
    ], [text[s:e] for s, e in chunk_spans(text, size=35)]
    print("ok")
