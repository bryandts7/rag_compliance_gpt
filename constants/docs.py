import ast
from  langchain.schema import Document
import json
from typing import Iterable
from database.database import rekam_jejak_docstore, ketentuan_terkait_docstore

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

def str_to_dict(s):
    # Replace the ObjectId string with a placeholder
    s = s.replace("ObjectId(", "").replace(")", "")

    # Use ast.literal_eval to safely evaluate the string
    d = ast.literal_eval(s)
    
    # Manually convert the _id field back to an ObjectId
    if '_id' in d:
        del d['_id']
    
    return d

def db_to_docs(docs):
    res = []
    for doc in docs:
        dic = str_to_dict(doc.page_content)
        lang_doc = Document(page_content=dic['page_content'], metadata=dic['metadata'])
        res.append(lang_doc)    
    return res

# REKAM_DOCSTORE =  rekam_jejak_docstore().load()
# KETENTUAN_DOCSTORE = ketentuan_terkait_docstore().load()

# REKAM_DOCS = db_to_docs(REKAM_DOCSTORE)
# KETENTUAN_DOCS = db_to_docs(KETENTUAN_DOCSTORE)

REKAM_DOCS = load_docs_from_jsonl(r"constants/rekam_jejak_docs_langchain.jsonl")
KETENTUAN_DOCS = load_docs_from_jsonl(r"constants/ketentuan_terkait_docs_langchain.jsonl")