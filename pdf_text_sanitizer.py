import os
import re
import unicodedata


def split_pages(text: str) -> list[str]:
    """
    Splits text into pages using common page marker format.
    """
    pages = re.split(r'-{5,}\s*Page\s+\d+\s*-{5,}', text)
    return [page.strip() for page in pages if page.strip()]


def split_columns_from_txt(text: str) -> str:
    left_col = []
    right_col = []
    lines = text.splitlines()

    # Count lines that look like 2-column (split by 5+ spaces or tabs)
    split_lines = 0
    total_lines = 0
    for line in lines:
        if line.strip() == "":
            continue
        total_lines += 1
        parts = re.split(r'\t+|\s{5,}', line)
        if len(parts) == 2:
            left = parts[0].strip()
            right = parts[1].strip()
            if not left or not right:
                # just layout for alignment, no real columns
                continue
            # Ignore if right contains no letters. Probably a page index.
            if not re.search(r'[a-zA-Z]', right):
                continue
            split_lines += 1

    is_two_column = split_lines / total_lines >= 0.6 if total_lines > 0 else False

    if is_two_column:
        for line in lines:
            parts = re.split(r'\t+|\s{5,}', line)
            if len(parts) == 2:
                left_col.append(parts[0].strip())
                right_col.append(parts[1].strip())
            elif len(parts) == 1:
                left_col.append(parts[0].strip())
        return "\n".join(left_col + [""] + right_col)
    else:
        return text


def sanitize_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sanitized = []
    buffer = ""

    # Remove double (or more) spaces inside line
    lines = [re.sub(r'\s{2,}', ' ', line) for line in lines]

    sentence_end = re.compile(r'[.!?]["\')\]]?$')

    for i, line in enumerate(lines):
        # Skip numeric-only lines (e.g. page numbers)
        if re.fullmatch(r'\d+', line):
            continue

        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        if buffer:
            if len(buffer) > 1 and buffer[-1] == '-' and buffer[-2].isalpha():
                buffer = buffer[0:-1]
                buffer += line
            else:
                buffer += " " + line
        else:
            buffer = line

        if sentence_end.search(line):
            # Ends in punctuation → end of paragraph
            sanitized.append(buffer.strip())
            sanitized.append("")
            buffer = ""
        elif line.startswith("---"):
            # new page marker probably
            sanitized.append(buffer.strip())
            sanitized.append("")
            buffer = ""
        elif next_line and not sentence_end.search(line):
            if re.match(r'^[A-Z]', next_line):
                # Next line starts with capital → section break
                sanitized.append(buffer.strip())
                sanitized.append("")
                buffer = ""
            elif next_line.startswith("---"):
                # new page marker on next line probably
                sanitized.append(buffer.strip())
                sanitized.append("")
                buffer = ""
            else:
                if line.endswith('-'):
                    pass
                # Next line starts with lowercase → merge (continue buffering)
                continue
        else:
            # Last line fallback
            sanitized.append(buffer.strip())
            buffer = ""

    return "\n".join(sanitized)


def remove_malformed_diacritics(text: str) -> str:
    """
    Removes malformed standalone diacritic characters often caused by OCR or bad encoding.
    Only removes when followed by a letter.
    """
    malformed_diacritics = [
        '\u00A8',  # ¨ diaeresis
        '\u00B4',  # ´ acute accent
        '\u0060',  # ` grave accent (ASCII backtick)
        '\u005E',  # ^ ASCII caret
        '\u02C6',  # ˆ modifier letter circumflex (the one in your example)
        '\u02C7',  # ˇ caron
        '\u02D8',  # ˘ breve
        '\u00B8',  # ¸ cedilla
        '\u007E',  # ~ ASCII tilde
    ]

    # Remove only when followed by a word character (letter or digit)
    return re.sub(
        rf"[{''.join(re.escape(ch) for ch in malformed_diacritics)}](?=\w)",
        '',
        text
    )


def remove_diacritics(text: str) -> str:
    """
    Removes both malformed and proper diacritics while preserving all other characters.
    """
    text = remove_malformed_diacritics(text)
    text = unicodedata.normalize("NFD", text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", text)


def fix_all_txt_files(input_folder: str, output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".txt"):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with open(input_path, encoding="utf-8") as f:
                raw_text = f.read()

            text_without_nbsp = raw_text.replace('\u00A0', ' ')
            text_without_diacritics = remove_diacritics(text_without_nbsp)
            pages = split_pages(text_without_diacritics)
            cleaned_pages = [split_columns_from_txt(page) for page in pages]
            single_column_text = "\n\n".join(cleaned_pages)
            sanitized_text = sanitize_lines(single_column_text)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(sanitized_text)

            print(f"✔ Processed {filename}")
        except Exception as e:
            print(f"✘ Failed to process {filename}: {e}")


if __name__ == "__main__":
    fix_all_txt_files("source_texts", "source_texts_cleaned")
    fix_all_txt_files("doc_to_check", "doc_to_check_cleaned")
