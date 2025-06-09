import re
import json
from pathlib import Path
from typing import List


def contains_letters(s: str) -> bool:
    return bool(re.search(r'[a-zA-Z]', s))


def load_text(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def split_into_paragraphs(text: str) -> List[str]:
    return re.split(r"\n{2,}", text)


def find_claims(text: str, citations: List[str]) -> dict:
    paragraphs = split_into_paragraphs(text)
    claims = {x: [] for x in citations}
    prev_par = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        for citation in citations:
            index = paragraph.find(citation)
            if index == -1:
                continue
            if not contains_letters(paragraph[:index]) and not contains_letters(paragraph[len(citation):]):
                # accidental break between paragraph and citation
                claims[citation].append(f"{prev_par} {paragraph}")
            else:
                claims[citation].append(paragraph)
        prev_par = paragraph
    return claims


def save_claims(claims: dict, output_file: str):
    Path(output_file).write_text(json.dumps(claims, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    text_path = "doc_to_check/doc_to_check.txt"
    citation_map_path = "doc_to_check/citation_map.json"

    text = load_text(text_path)
    citation_map = json.loads(Path(citation_map_path).read_text(encoding="utf-8"))
    citations = list(citation_map.keys())

    claims = find_claims(text, citations)
    save_claims(claims, "doc_to_check/claims.json")
