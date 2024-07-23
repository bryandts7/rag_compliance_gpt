from langchain.vectorstores.redis import Redis as RedisVectorStore
from langchain_community.document_loaders.mongodb import MongodbLoader

from utils.azure_openai import azure_embeddings

import time
import os
import dotenv
dotenv.load_dotenv()

embeddings = azure_embeddings()

# Redis Database
redis_url = os.environ.get("REDIS_URI")
print(redis_url)

# MongoDB Database
mongo_uri = os.environ.get("MONGO_URI")

def rekam_jejak_vector():
    rekam_jejak_vector = RedisVectorStore.from_existing_index(
                            embedding = embeddings,
                            index_name = "sikepo-rekam-jejak",
                            redis_url = redis_url,
                            schema = r"constants/redis_schema_sikepo_rekam.yaml",
                        )
    return rekam_jejak_vector

def rekam_jejak_docstore():
    REKAM_JEJAK_DOCSTORE = MongodbLoader(
    connection_string= mongo_uri, # type: ignore
    db_name="docstore",
    collection_name="rekam_jejak",
    )
    return REKAM_JEJAK_DOCSTORE

def ketentuan_terkait_vector():
    ketentuan_terkait_vector = RedisVectorStore.from_existing_index(
                            embedding = embeddings,
                            index_name = "sikepo-ketentuan-terkait",
                            redis_url = redis_url,
                            schema = r"constants/redis_schema_sikepo_ketentuan.yaml",
                        )
    return ketentuan_terkait_vector

def ketentuan_terkait_docstore():
    KETENTUAN_TERKAIT_DOCSTORE = MongodbLoader(
    connection_string= mongo_uri, # type: ignore
    db_name="docstore",
    collection_name="ketentuan_terkait",
    )
    return KETENTUAN_TERKAIT_DOCSTORE