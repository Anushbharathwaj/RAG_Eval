import pymupdf as fitz

from typing import List
from pathlib import Path
import re

#Layout-Based Chunking
def extract_paragraphs(pdf_path: str| Path) -> List[str]:
    """This function takes pdf path and return a array of its paragraphs"""

    doc = fitz.open(pdf_path)

    para  = []

    for page , content in enumerate(doc):
        blocks =  content.get_text("blocks")
        blocks = sorted(blocks, key = lambda b : (b[1],b[0]))

        for block in blocks:
            text = block[4]

            text = re.sub(r"\r+"," ",text).strip()

            if text:
                para.append(text)

    return para

#print(extract_paragraphs("pdfs/Penguins_ACL.pdf"))

