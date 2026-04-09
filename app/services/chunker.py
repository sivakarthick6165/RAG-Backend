from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
