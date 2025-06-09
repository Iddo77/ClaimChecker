import re

GAP = "[...]"


def normalize_text(txt: str) -> str:
    return re.sub(r'[^a-z]+', '', txt.lower())


def reconstruct_from_trigrams(paper: str, quote: str) -> list[str]:
    normalized_paper = normalize_text(paper)
    words = quote.split()
    n = len(words)
    parts = []
    i = 0
    last_norm_pos = 0  # where to continue searching from

    while i < n - 2:
        found = False
        longest_match = None
        match_pos = -1
        norm_chunk_size = 0
        gap_size = 0

        for j in range(i + 3, n + 1):  # try chunks from 3 words up
            chunk = " ".join(words[i:j])
            norm_chunk = normalize_text(chunk)
            idx = normalized_paper.find(norm_chunk, last_norm_pos)

            if idx != -1:
                found = True
                longest_match = chunk
                match_pos = idx
                gap_size = idx - last_norm_pos
                norm_chunk_size = len(norm_chunk)
            else:
                break  # stop at first failure

        if found:
            if len(parts) and gap_size > 0:
                parts.append(gap_size)
            parts.append(longest_match)
            last_norm_pos = match_pos + norm_chunk_size
            i += len(longest_match.split())
        else:
            i += 1

    # Final tail
    if i < n:
        tail_chunk = " ".join(words[i:])
        norm_tail = normalize_text(tail_chunk)
        idx = normalized_paper.find(norm_tail, last_norm_pos)

        if idx != -1:
            gap_size = idx - last_norm_pos
            if len(parts) and gap_size > 0:
                parts.append(gap_size)
            parts.append(tail_chunk)

    return parts


def validate_gaps(parts: list[str | int], max_norm_gap: int = 75) -> bool:
    for idx, part in enumerate(parts):
        if isinstance(part, int) and part > max_norm_gap:
            left_size = len(parts[idx - 1]) if idx > 0 else 0
            right_size = len(parts[idx + 1]) if idx < len(parts) - 1 else 0
            if part > left_size or part > right_size:
                return False
    return True


def reconstruct_parts(parts: list[str]) -> str:
    parts = [part for part in parts if isinstance(part, str)]
    return ' '.join(parts)


def validate_and_reconstruct(paper_text, quote):
    parts = reconstruct_from_trigrams(paper_text, quote)
    if validate_gaps(parts):
        return reconstruct_parts(parts)
    else:
        print(parts)
        return None

