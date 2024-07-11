from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from utils.azure_openai import azure_embeddings

import time
import os
import dotenv
dotenv.load_dotenv()

# initialize MongoDB python client
embeddings = azure_embeddings()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

def rekam_jejak_vector():
    index_name = "rekam-jejak"
    index = pc.Index(index_name)
    rekam_jejak_vector = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return rekam_jejak_vector

def ketentuan_terkait_vector():
    index_name = "ketentuan-terkait"
    index = pc.Index(index_name)
    ketentuan_terkait_vector = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    return ketentuan_terkait_vector