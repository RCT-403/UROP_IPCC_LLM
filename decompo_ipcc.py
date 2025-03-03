'''we have 4 level of headers 
level 0: title (AR6 Group 1)
level 1: 9 
level 2: 9.1
level 3: 9.1.1 (bolded)
level 4: 9.1.1.1 (newlines chars)
level 5: 9.1.1.1.1 (newlines and italic chars)
'''
from typing import Dict, List, Tuple
import pymupdf4llm
import pymupdf
import time

headers_pattern = { # level: (start_pattern)
    1: (), # chapter
    2: ("\n\n ##### "), 
    3: ("\n\n**{}.{}.{}**".format(i, j, k) for i in range(1, 10) for j in range(1, 10) for k in range(1, 10)),
    4: ("\n\n{}.{}.{}.{}".format(i, j, k, l) for i in range(1, 10) for j in range(1, 10) for k in range(1, 10) for l in range(10, 20)),
    5: ("\n\n{}.{}.{}.{}.{}".format(i, j, k, l, m) for i in range(1, 10) for j in range(1, 10) for k in range(1, 10) for l in range(10, 20) for m in range(10, 20))
}

'''
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
'''
      
class Section:
    def __init__(self, title, content="", parent=None):
        self.title = title
        self.content = content
        self.parent = parent
        self.children = []
        self._index = None

    def add_child(self, title):
        child = Section(title, parent=self)
        self.children.append(child)
        return child

    @property
    def index(self):
        if self.parent is None:
            return "0"
        if self._index is None:
            parent_index = self.parent.index if self.parent else ""
            sibling_position = str(self.parent.children.index(self) + 1)
            self._index = f"{parent_index}.{sibling_position}" if parent_index != "0" else sibling_position
        return self._index

    def to_dict(self):
        return {
            "index": self.index,
            "title": self.title,
            "content": self.content,
            "children": [child.to_dict() for child in self.children]
        }
    
def extract_toc(report: Section, pdf_path: str):
    doc = pymupdf.open(pdf_path)
    title = doc.metadata["title"]
    toc = doc.get_toc()
    for level, title, page_num in toc:
        if level == 0:
            continue
        parent = report
        for i in range(level - 1):
            parent = parent.children[-1] # get the last child
        parent.add_child(title) # add the title to the last child
    
def extract_content(report: Section, pdf_path: str, level: int):
    '''Read through the pdf and extract the content for each level x section'''
    pass # to-do
    
def main():
    # pdf_path = "./data/IPCC_AR6_WGI_Chapter09 Section 1 to 2.pdf"
    pdf_path = "./data/IPCC AR6 Chapter 2 Climate System.pdf"

    report = Section("IPCC AR6 WGI")
    extract_toc(report, pdf_path)

    print(report.to_dict())

if __name__ == "__main__":
    main()