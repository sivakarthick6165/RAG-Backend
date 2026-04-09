import os
import fitz  # PyMuPDF
import pandas as pd
import json
from typing import List

class FileParser:
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    @staticmethod
    def parse_excel(file_path: str) -> str:
        df = pd.read_excel(file_path)
        return df.to_string()

    @staticmethod
    def parse_json(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)

    @staticmethod
    def parse_txt(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @classmethod
    def extract_text(cls, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return cls.parse_pdf(file_path)
        elif ext in ['.xlsx', '.xls']:
            return cls.parse_excel(file_path)
        elif ext == '.json':
            return cls.parse_json(file_path)
        elif ext == '.txt':
            return cls.parse_txt(file_path)
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            return df.to_string()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
