from collections import defaultdict
import json


def summarize_citations(data: dict) -> dict:
    summary = {}

    for citation, entries in data.items():
        total = len(entries)
        low = sum(1 for e in entries if e["confidence"] == "LOW")
        med = sum(1 for e in entries if e["confidence"] == "MEDIUM")
        high = sum(1 for e in entries if e["confidence"] == "HIGH")
        consistent = sum(1 for e in entries if e.get("is_consistent"))
        missing = sum(1 for e in entries if not e.get("quote", "").strip())

        valid_cosines = [e["cosine"] for e in entries if isinstance(e.get("cosine"), (int, float))]
        avg_cosine = round(sum(valid_cosines) / len(valid_cosines), 3) if valid_cosines else 0.0

        summary[citation] = {
            "total": total,
            "LOW": low,
            "MEDIUM": med,
            "HIGH": high,
            "consistent": consistent,
            "missing": missing,
            "avg_cosine": avg_cosine
        }

    return summary


if __name__ == "__main__":
    with open("doc_to_check/validated_claims.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    stats = summarize_citations(data)

    with open("doc_to_check/citation_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
