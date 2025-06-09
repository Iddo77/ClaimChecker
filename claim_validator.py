import openai
import json
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv(override=True)
CONFIDENCE_MAP = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.8}


def get_embedding(text: str) -> list[float]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding


def validate_claims(data: dict) -> dict:
    validated = dict()

    for citation, entries in data.items():
        validated[citation] = []
        for entry in entries:
            paragraph = entry["paragraph"]
            quote = entry["quote"]
            declared_conf = entry["confidence"]

            emb_par = get_embedding(paragraph)
            if quote:
                emb_quote = get_embedding(quote)
                cos_sim = float(cosine_similarity([emb_par], [emb_quote])[0][0])

                if cos_sim > CONFIDENCE_MAP['HIGH']:
                    cosine_conf = 'HIGH'
                elif cos_sim > CONFIDENCE_MAP['MEDIUM']:
                    cosine_conf = 'MEDIUM'
                else:
                    cosine_conf = 'LOW'

                is_consistent = cosine_conf == declared_conf

                score = 0.5 * cos_sim + 0.5 * CONFIDENCE_MAP.get(declared_conf.upper(), 0.3)
            else:
                cos_sim = 0.0
                is_consistent = True
                score = 0.0

            validated[citation].append({
                "paragraph": paragraph,
                "quote": quote,
                "confidence": declared_conf,
                "cosine": round(cos_sim, 3),
                "is_consistent": bool(is_consistent),
                "score": round(score, 3)
            })

    return validated


if __name__ == "__main__":
    with open("doc_to_check/check_citations.json", "r", encoding="utf-8") as file:
        check_citations_map = json.load(file)
    result = validate_claims(check_citations_map)
    with open("doc_to_check/validated_claims.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
