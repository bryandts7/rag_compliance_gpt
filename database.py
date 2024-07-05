from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch

from azure_config import azure_embeddings

import os
import dotenv
dotenv.load_dotenv()

MONGODB_ATLAS_CLUSTER_URI  = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
DB_NAME = "langchain_db"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
embeddings = azure_embeddings()

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
    COLLECTION_NAME = "ketentuan_terkait"
    ketentuan_terkait_vector = MongoDBAtlasVectorSearch.from_connection_string(
        MONGODB_ATLAS_CLUSTER_URI, # type: ignore
        DB_NAME + "." + COLLECTION_NAME,
        embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    return ketentuan_terkait_vector