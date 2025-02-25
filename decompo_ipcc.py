'''we have 4 level of headers 
level 1: 9 
level 2: 9.1
level 3: 9.1.1 (bolded)
level 4: 9.1.1.1 (newlines chars)
'''
from typing import Dict, List, Tuple
import pymupdf4llm
import pymupdf
import time

headers_pattern_start = { # level: (start_pattern)
    1: (),
    2: ("\n##### {}.{}".format(i, j) for i in range(1, 10) for j in range(1, 10)),
    3: ("\n\n**{}.{}.{}**".format(i, j, k) for i in range(1, 10) for j in range(1, 10) for k in range(1, 10)),
    4: ("\n\n{}.{}.{}.{}".format(i, j, k, l) for i in range(1, 10) for j in range(1, 10) for k in range(1, 10) for l in range(1, 10))
}

def extract_toc(pdf_path) -> Dict:
    doc = pymupdf.open(pdf_path)
    return doc.get_toc() #we have level and title and one useless property # to-do: "clean" the titles

def extract_content_by_section(pdf_path):
    start = time.time()
    md_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=False, force_text=True)
    time_taken = time.time() - start
    print(f"Time taken: {time_taken / 60:.2f} minutes")

    # search for the section titles and consequently extract the content
    sections: List[str] = []
    for pattern in headers_pattern_start[4]: # consider only the innermost header
        md_text = md_text.replace(pattern, f"@*@ {pattern}")
    sections = md_text.split("@*@")

    formatted_sections: List[Tuple[str, str]] = [] # (title, content)
    for i, section in enumerate(sections):
        _, title, content = section.split("\n\n", 2) # to-do: for header 2 and 3, this doesn't work
        formatted_sections.append((title, content))

    return None

def main():
    pdf_path = "./data/IPCC_AR6_WGI_Chapter09 Section 1 to 2.pdf"

    toc = extract_toc(pdf_path)
    extract_content_by_section(pdf_path)

    print("End")

if __name__ == "__main__":
    main()