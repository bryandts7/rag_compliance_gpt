from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from azure_config import azure_embeddings

import time
import os
import dotenv
dotenv.load_dotenv()

MONGODB_ATLAS_CLUSTER_URI  = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
DB_NAME = "langchain_db"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
embeddings = azure_embeddings()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "ketentuan-terkait"  # change if desired
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
index = pc.Index(index_name)

def rekam_jejak_vector():
    COLLECTION_NAME = "rekam_jejak"
    rekam_jejak_vector = MongoDBAtlasVectorSearch.from_connection_string(
        MONGODB_ATLAS_CLUSTER_URI, # type: ignore
        DB_NAME + "." + COLLECTION_NAME,
        embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    return rekam_jejak_vector

def ketentuan_terkait_vector():
    # COLLECTION_NAME = "ketentuan_terkait"
    # ketentuan_terkait_vector = MongoDBAtlasVectorSearch.from_connection_string(
    #     MONGODB_ATLAS_CLUSTER_URI, # type: ignore
    #     DB_NAME + "." + COLLECTION_NAME,
    #     embeddings,
    #     index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    # )
    ketentuan_terkait_vector = PineconeVectorStore(index_name=index_name, embedding=embeddings)


    return ketentuan_terkait_vector