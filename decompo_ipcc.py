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
from langchain_text_splitters import MarkdownHeaderTextSplitter
import re 
import json

'''deprecated code
headers_pattern = { # level: (start_pattern)
    1: (), # chapter
    2: ("\n\n ##### "), 
    3: ("\n\n**{}.{}.{}**".format(i, j, k) for i in range(1, 10) for j in range(1, 10) for k in range(1, 10)),
    4: ("\n\n{}.{}.{}.{}".format(i, j, k, l) for i in range(1, 10) for j in range(1, 10) for k in range(1, 10) for l in range(10, 20)),
    5: ("\n\n{}.{}.{}.{}.{}".format(i, j, k, l, m) for i in range(1, 10) for j in range(1, 10) for k in range(1, 10) for l in range(10, 20) for m in range(10, 20))
}

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
    
    def add_child_section(self, new_section: "Section"):
        print("Adding child section:", new_section.title, "to parent:", self.title)
        new_section.parent = self
        self.children.append(new_section)

    @property
    def index(self):
        '''
        if self.parent is None:
            return "0"
        if self._index is None:
            parent_index = self.parent.index if self.parent else ""
            sibling_position = str(self.parent.children.index(self) + 1)
            self._index = f"{parent_index}.{sibling_position}" if parent_index != "0" else sibling_position
        return self._index
        '''

        pattern = re.compile(r'(\d+\.\d+(\.\d+)*)')
        match = pattern.match(self.title)
        if match:
            return match.group(1)
        return None

    def to_dict(self):
        return {
            "index": self.index,
            "title": self.title,
            "content": self.content,
            "parent": self.parent.index if self.parent else None,
            "children": [child.to_dict() for child in self.children]
        }
            
def extract_toc(report: Section, pdf_path: str):
    doc = pymupdf.open(pdf_path)
    title = doc.metadata["title"]
    report.add_child(title) # add the title to the report

    toc = doc.get_toc()
    for level, title, page_num in toc:
        if level == 0:
            continue
        parent = report.children[-1] # !! for now, the last child is the report title
        for i in range(level - 1):
            parent = parent.children[-1] # get the last child
        parent.add_child(title) # add the title to the last child

def find_imm_parent(current_section: Section, report: Section):
    '''Find the immediate parent of the current section
    each in the X.X.X.X.X represent a index, 
    don't need the last one as it represents current section'''
    parent_section = report
    for idx in current_section.title.split('.')[:-1]:
        if parent_section.parent is None:
            parent_section = parent_section.children[0] # for now, assume we hv only one chapter
        elif parent_section.children[0] == "Executive Summary":
            parent_section = parent_section.children[int(idx) - 1]
        else:
            for cand in parent_section.children: # avoid accessing boxes or other special session
                splits = cand.title.split('.')
                if len(splits) > 1 and splits[-1][0] == idx:
                    parent_section = cand
                    break
        print("Traverse to section:", parent_section.title)
    return parent_section

def extract_content(report: Section, pdf_path: str):
    '''Read through the pdf and extract the content for each level x section'''
    doc = pymupdf.open(pdf_path)
    data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, force_text=True) #, pages=list(range(1, 50))

    # search for the section titles and consequently extract the content
    header_pattern = re.compile(r'(\d+\.\d+\.\d+\.\d+(\.\d+)*)')

    current_section = None

    for page in data: # avoid memory issues
        mdtext = page["text"]
        for line in mdtext.split("\n"): #could be slow
            match = header_pattern.match(line)
            if match: 
                # If there's a current section, add it to the parent section
                if current_section:
                    parent_section = find_imm_parent(current_section, report)
                    parent_section.add_child_section(current_section)
                    current_section = None
                
                # Start a new section
                print("\nCreate a new section:", line)
                title = line.strip()
                current_section = Section(title)
                           
            elif current_section:
                # Add content to the current section
                current_section.content += line + "\n"
        
        # Add the last section to the parent section
        if current_section:
            parent_section = find_imm_parent(current_section, report)
            parent_section.add_child_section(current_section)
            current_section = None

def search_for_content(report: Section, section_to_search: str):
    '''Search for a specific content in the report. Return the title and content of the section
    section_to_search: the index of the section to search for, e.g. 2.3.1

    remark: this function is similar to find_imm_parent'''
    sfs = section_to_search.split('.')
    cur_section = report
    for idx in sfs:
        if cur_section.parent is None:
            cur_section = cur_section.children[0]
        elif cur_section.children[0] == "Executive Summary":
            cur_section = cur_section.children[int(idx) - 1]
        else:
            for cand in cur_section.children:
                splits = cand.title.split('.')
                if len(splits) > 1 and splits[-1][0] == idx:
                    cur_section = cand
                    break
    return cur_section.title, cur_section.content

def build_report(pdf_path: str):
    report = Section("IPCC AR6 WGI")
    extract_toc(report, pdf_path)
    extract_content(report, pdf_path)
    return report

def main(): 
    pdf_path = "./data/IPCC AR6 Chapter 2 Climate System.pdf"

    time_start = time.time()
    report = build_report(pdf_path)
    print(f"Time taken: {(time.time() - time_start) / 60:.2f} minutes")
    
    # print(json.dumps(report.to_dict(), indent=4))

    title_a, title_b = search_for_content(report, "2.3.1.1.1")
    print("Content found:")
    print(title_a)
    print(title_b)

    print("End of main()")

if __name__ == "__main__":
    main()