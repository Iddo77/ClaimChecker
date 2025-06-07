import json
import re
import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def extract_json_block(text: str) -> str:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    raise ValueError("No JSON block found in the response")


def map_citations_to_references(paper_txt: str, references_txt: str) -> dict:
    references = references_txt.splitlines()

    prompt = f"""
Given the full text of a scientific paper and a list of reference entries, identify all in-text citations
(e.g., (Author et al., 2024), [1], etc.) and match each to the best corresponding line from the references list.

Paper Text (between %%%):
%%%
{paper_txt}
%%%

References (between %%%):
%%%
{references}
%%%

Output a JSON dictionary where each citation in the text maps to the best matching line from the references.
Only map UNIQUE citations. Map ALL citations. Pay attention to diacritics in references

"""

    example = """
    
    Example (not real):
```json
{
    "Smith et al. (2023)": "Smith, J., Doe, A., and Lee, K. (2023). AI in education. Journal of Learning Tech, 12(3), 45–52."
}
```
    """

    prompt += example

    messages = [
        {"role": "system", "content": "You are an expert at parsing and mapping scientific references."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0
    )

    max_retries = 2
    i = 0
    json_obj = None
    while i < max_retries:
        i += 1
        json_block = extract_json_block(response.choices[0].message.content)
        json_obj = json.loads(json_block)
        errors = validate_citation_map(json_obj, paper_txt, references_txt)
        if not errors:
            return json_obj
        else:
            messages.append({"role": "assistant", "content": json_block})
            error_msg = f"""
            I found the following problems:
            {errors}
            
            Please fix them. Make sure to be EXACT in your response, because the validation is automatic. Also take diacritics in citations into account.
            """
            messages.append({"role": "user", "content": error_msg})

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0
            )

    return json_obj if json_obj else {}


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

def citation_map_to_file_map(citation_map_path: str, references_path: str, sources_dir: str, output_path: str):
    with open(citation_map_path, "r", encoding="utf-8") as f:
        citation_map = json.load(f)

    with open(references_path, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f if line.strip()]

    filenames = os.listdir(sources_dir)
    file_map = {}

    ref_index_to_file = {}
    for fname in filenames:
        match = re.match(r"\[(\d+)]", fname)
        if match:
            index = int(match.group(1))
            ref_index_to_file[index] = fname

    for citation, ref_line in citation_map.items():
        try:
            ref_index = references.index(ref_line) + 1  # 1-based index
            file_name = ref_index_to_file.get(ref_index)
            file_map[citation] = file_name
        except ValueError:
            print("Reference not found: ", ref_line)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(file_map, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    with open("doc_to_check/doc_to_check.txt", "r", encoding="utf-8") as f:
        paper_txt = f.read()

    with open("sources/references.txt", "r", encoding="utf-8") as f:
        references_txt = f.read()

    citation_map = map_citations_to_references(paper_txt, references_txt)
    with open("doc_to_check/citation_map.json", "w", encoding="utf-8") as f:
        json.dump(citation_map, f, ensure_ascii=False, indent=2)

    citation_map_to_file_map(
        citation_map_path="doc_to_check/citation_map.json",
        references_path="sources/references.txt",
        sources_dir="source_texts",
        output_path="doc_to_check/file_map.json"
    )
