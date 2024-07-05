from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from bson import ObjectId
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import List, Sequence, Union
from langchain_core.messages import BaseMessage

import string
import pandas as pd
import os
import dotenv
import os
import random

dotenv.load_dotenv()

store = {}

class CustomChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history.

    Stores messages in an in memory list.
    """

    messages: List[BaseMessage] = Field(default_factory=list)
    max_messages: int = 5  # Set the limit (K) of messages to keep

    async def aget_messages(self) -> List[BaseMessage]:
        return self.messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)
        # Keep last k messages
        self.messages = self.messages[-self.max_messages:]


    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the store"""
        self.add_messages(messages)

    def clear(self) -> None:
        self.messages = []

    async def aclear(self) -> None:
        self.clear()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = CustomChatMessageHistory()
    return store[session_id]


# Configuration
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="embedding-ada-crayon",
    openai_api_version=api_version,
    api_key=api_key,
    azure_endpoint=endpoint,
)

llm = AzureChatOpenAI(
    azure_deployment="gpt-35-crayon",
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=endpoint,
    temperature=0
    # other params...
)

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["rekam_jejak", "ketentuan_terkait", ] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

structured_llm = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to the appropriate data source.
1.) User Inquiry about "Peraturan" or "Ketentuan" Status:
Criteria: If the user question asks about the relevance, modification, or history of "peraturan" or "ketentuan" (e.g., "Is this regulation still relevant?", "Has this rule been modified?", or any query related to "rekam jejak"),
Action: Return 'rekam_jejak'.

2.) User Inquiry for Detailed Explanation or Understanding:
Criteria: If the user question asks for detailed explanations, meanings of the regulations, or any queries unrelated to "rekam jejak" (e.g., "What does this regulation mean?", "Can you explain this rule in detail?", or any other unrelated questions),
Action: Return 'ketentuan_terkait'.

"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        # MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

def get_string(route):
    return route.datasource

# Define router 
router = prompt | structured_llm | RunnableLambda(get_string)

MONGODB_ATLAS_CLUSTER_URI  = os.getenv("MONGODB_ATLAS_CLUSTER_URI")

# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = "langchain_db"
COLLECTION_NAME = "rekam_jejak"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
rekam_jejak_vector = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_ATLAS_CLUSTER_URI,
    DB_NAME + "." + COLLECTION_NAME,
    embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)


COLLECTION_NAME = "ketentuan_terkait"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
ketentuan_terkait_vector = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_ATLAS_CLUSTER_URI,
    DB_NAME + "." + COLLECTION_NAME,
    embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

rekam_jejak_retriever = rekam_jejak_vector.as_retriever()
ketentuan_terkait_retriever = ketentuan_terkait_vector.as_retriever()

# RAG-Fusion: Related
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)
generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)


# Convert ObjectId to string before serialization
def convert_objectid_to_string(doc):
    if '_id' in doc.metadata:
        doc.metadata['_id'] = str(doc.metadata['_id'])
    return doc

# Convert string back to ObjectId after deserialization
def convert_string_to_objectid(doc):
    if '_id' in doc.metadata:
        doc.metadata['_id'] = ObjectId(doc.metadata['_id'])
        del doc.metadata['embedding']
    return doc

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(convert_objectid_to_string(doc))
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (convert_string_to_objectid(loads(doc)), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

def choose_retriever(result):
    print(result)
    if result["result"] == "rekam_jejak":
        return retrieval_rekam_jejak_chain_rag_fusion
    elif result["result"] == "ketentuan_terkait":
        return retrieval_ketentuan_terkait_chain_rag_fusion

retrieval_rekam_jejak_chain_rag_fusion = generate_queries | rekam_jejak_retriever.map() | reciprocal_rank_fusion
retrieval_ketentuan_terkait_chain_rag_fusion = generate_queries | ketentuan_terkait_retriever.map() | reciprocal_rank_fusion


# System prompt for RAG
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the answer concise."
    "Please write your answer ONLY in INDONESIAN."

    "Jika sebuah peraturan A mencabut atau mengubah peraturan B, C, D"
    "maka peraturan B, C, D tersebut mungkin saja sudah tidak berlaku"
    
    "Jika ada yang bertanya mengenai aturan masih berlaku atau masih relevan, "
    "jawab 'Iya' ketika peraturan tersebut tidak diubah dan tidak dicabut peraturan lain dan jelaskan mengapa demikian."
    "Sebaliknya, jawab 'Mungkin tidak berlaku' ketika ada peraturan lain yang mengubah atau mencabut peraturan tersebut. "
    "Tambahkan peraturan apa yang mencabut atau mengubah jika jawaban nya 'Mungkin tidak berlaku' "
    
    # "Please also mention the 'Nomor Ketentuan' and 'Ketentuan' from the metadata in a subtle way "
    # "such that you can add more context to answer the question. "
    "\n\n"
    "{context}"
)

# prompt = ChatPromptTemplate.from_template(template)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)    

context_chain = {
    "result":router,
    "question": itemgetter("question"),
    "history" : itemgetter("history")
    } | RunnablePassthrough() | {"context": RunnableLambda(choose_retriever), "question": itemgetter("question"), "history" : itemgetter("history") }


full_chain = context_chain | prompt | llm | StrOutputParser()

full_chain_with_message_history = RunnableWithMessageHistory(
    full_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# Function to invoke the full chain
def caller(message, sess_id):
    print(sess_id)
    print("Store:", store)
    response = full_chain_with_message_history.invoke(
        {"question": message},
        config={"configurable": {"session_id": sess_id}})
    return response


