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

        low_entries = [e for e in entries if e["confidence"] == "LOW" and isinstance(e.get("cosine"), (int, float))]
        medium_entries = [e for e in entries if e["confidence"] == "MEDIUM" and isinstance(e.get("cosine"), (int, float))]
        high_entries = [e for e in entries if e["confidence"] == "HIGH" and isinstance(e.get("cosine"), (int, float))]

        avg_cosine_low = round(sum(e["cosine"] for e in low_entries) / len(low_entries), 3) if low_entries else 0.0
        avg_cosine_medium = round(sum(e["cosine"] for e in medium_entries) / len(medium_entries), 3) if medium_entries else 0.0
        avg_cosine_high = round(sum(e["cosine"] for e in high_entries) / len(high_entries), 3) if high_entries else 0.0

        summary[citation] = {
            "total": total,
            "LOW": low,
            "MEDIUM": med,
            "HIGH": high,
            "consistent": consistent,
            "missing": missing,
            "avg_cosine": round((avg_cosine_low + avg_cosine_medium + avg_cosine_high) / 3, 3),
            "avg_cosine_LOW": avg_cosine_low,
            "avg_cosine_MEDIUM": avg_cosine_medium,
            "avg_cosine_HIGH": avg_cosine_high
        }

    return summary


if __name__ == "__main__":
    with open("doc_to_check/validated_claims.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    stats = summarize_citations(data)

    with open("doc_to_check/citation_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
