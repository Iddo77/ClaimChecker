import json
import re
import os
from dotenv import load_dotenv
from openai import OpenAI

from text_validation import normalize_text, reconstruct_from_trigrams, validate_gaps, validate_and_reconstruct

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def extract_json_block(text: str) -> str:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    raise ValueError("No JSON block found in the response")


def paper_contains_text(paper: str, text: str) -> bool:
    return normalize_text(text) in normalize_text(paper)


def check_claim(citation: str, paragraph: str, paper_txt: str) -> dict:
    prompt = f"""
    Your task is to verify that a claim referring to a paper is actually grounded in the paper.

    You are given:
    1. The full PAPER TEXT (between the FIRST set of triple percent signs, %%%).
    2. A CITATION referring to that paper.
    3. A PARAGRAPH that contains a claim referencing that citation.

    IMPORTANT:
    - ONLY extract a quote from the PAPER TEXT.
    - DO NOT quote from the paragraph containing the claim.
    - Your quote will be checked by an automated string search on the PAPER TEXT. If it is not found there, your answer will be rejected.
    - If you cannot find any quote in the paper that supports the claim, respond with a valid quote from the paper that is MOST LIKELY related, and mark the confidence as "LOW".
    - If you mistakenly return text from the paragraph containing the claim, your response will be invalid.
    - The paper may contain mid-sentence interruptions by headers or layout artifacts, because the PDF was parsed to plain text. When quoting such a sentence or paragraph, you must include the erroneous text exactly as it appears. Do not fix or remove layout errors — if your quote does not match the paper verbatim, it will be rejected by exact-match checking.

    PAPER TEXT (between %%%):
    %%%
    {paper_txt}
    %%%

    CITATION referring to this paper: 
    {citation}

    PARAGRAPH containing the claim (between %%%):
    %%%
    {paragraph}
    %%%

    Return the EXACT sentence or paragraph from the PAPER TEXT that best substantiates the claim made about it in the paragraph.

    Your output must be valid JSON in the following format:
    """

    example = """
```json
{
    "quote": "exact sentence from the PAPER TEXT that supports the claim."
    "confidence": "LOW|MEDIUM|HIGH"
}
```
    """

    prompt += example

    messages = [
        {"role": "system", "content": "You are an expert at verifying claims in scientific papers."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0
    )

    max_retries = 5
    i = 0

    json_obj = {"quote": "", "confidence": "LOW"}

    while i < max_retries:
        i += 1
        response_text = response.choices[0].message.content

        try:
            json_block = extract_json_block(response_text)
            json_obj = json.loads(json_block)
            response_text = json_obj['quote']
        except ValueError:
            continue

        text_found = paper_contains_text(paper_txt, response_text)

        if not text_found:
            reconstructed_text = validate_and_reconstruct(paper_txt, response_text)
            if reconstructed_text:
                json_obj['quote'] = reconstructed_text
                break

        if not text_found:
            messages.append({"role": "assistant", "content": response_text})

            if paper_contains_text(paragraph, response_text):
                error_msg = ("You returned an quote from the paragraph with the claim instead of from the paper. "
                             "I hope you realize this seriously jeopardizes are scientific project, as semantic similarity will be 100%. "
                             "Fix it and return the JSON with a quote from the paper instead nothing else.")
            else:
                error_msg = ("AUTOMATIC VERIFICATION FAILED: the quote is not found in the paper. "
                             "Fix your response and return the JSON with an EXACT quote from the PAPER TEXT and nothing else."
                             "\nMaybe it's an idea to reduce your quote in size so it's more likely the text is found, "
                             "despite errors. Only semantically meaningful parts are needed.")

            messages.append({"role": "user", "content": error_msg})
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.6
            )
        else:
            print(response_text)
            break

    if i == max_retries:
        json_obj = {"quote": "", "confidence": "LOW"}

    return json_obj


def validate_citation_map(citation_map_json, paper_txt: str, references_txt: str) -> list[str]:
    errors = []
    ref_lines = []
    for citation, ref_line in citation_map_json.items():
        ref_lines.append(ref_line)
        if citation not in paper_txt:
            errors.append(f"Citation not found in paper: {citation}")
        elif ref_line not in references_txt:
            errors.append(f"References not found in references: {ref_line}")
    for line in references_txt.splitlines():
        if line not in ref_lines:
            errors.append(f"Reference has no citation: {line}")

    return errors


def check_claims():
    with open("doc_to_check/claims.json", "r", encoding="utf-8") as file:
        claims_map = json.load(file)
    with open("doc_to_check/file_map.json", "r", encoding="utf-8") as file:
        file_map = json.load(file)

    checked_claims = dict()

    for citation, paragraphs in claims_map.items():
        claim_substantiations = []
        try:
            paper_file = file_map[citation]
            with open(f"source_texts_cleaned/{paper_file}", "r", encoding="utf-8") as file:
                paper_text = file.read()
            if len(paper_text) > 650000:
                paper_text = paper_text[:650000]  # cut off to fit context window
            for paragraph in paragraphs:
                result = check_claim(citation, paragraph, paper_text)
                result['paragraph'] = paragraph
                claim_substantiations.append(result)
        except Exception as e:
            print(e)

        checked_claims[citation] = claim_substantiations

    with open("doc_to_check/check_citations.json", "w", encoding="utf-8") as f:
        json.dump(checked_claims, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    check_claims()

