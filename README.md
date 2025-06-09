## STEPS TO USE

1. Place a file `references.txt` in the `sources` folder. Make sure it contains the references section from the paper, with one reference per line. Remove all diacritics by normalizing them.
2. Download the PDF for each reference and place them in the `sources` folder. Make sure to prefix the filename with [x] where x is the line number in `references.txt`.
3. Extract the text of each PDF and place them in the `source_texts` folder. This can be done with a PDF reader like Foxit Reader: File -> Save as -> *.txt
4. Extract the text of the document to check and save it to `doc_to_check/doc_to_check.txt`.
5. Run `pdf_text_sanitizer.py`. The texts will be cleaned and saved to `source_texts_cleaned`. Validate that it is correct. If not, it must be fixed manually.
6. Also, a folder `doc_to_check_cleaned` is created. Validate the `doc_to_check.txt` and move it to the folder `docx_to_check`, overwriting the original file.
5. Place the OpenAI API key in the .env file.
6. Run `citation_mapper.py`. Check the results exhaustively in the `doc_to_check` folder. All citations should be found and mapped to references as well as to files. Unfortunately, GPT4o often fails partly in this task, so you might have to try and run the code multiple times. Possibly, you can manually combine the results of multiple runs.
7. Run `citation_extractor.py`. A file `claims.json` will be created containing a mapping between a citation and all paragraphs in which it occurs.
8. Run `claim_checker.py`. A file `check_citations.json` is created containing a quote from the paper that should substantiate a claim made in a paragraph, together with a confidence.