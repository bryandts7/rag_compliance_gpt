from  langchain.schema import Document
import json
from typing import Iterable

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

REKAM_DOCS = load_docs_from_jsonl(r"constants/rekam_jejak_docs_langchain.jsonl")
KETENTUAN_DOCS = load_docs_from_jsonl(r"constants/ketentuan_terkait_docs_langchain.jsonl")