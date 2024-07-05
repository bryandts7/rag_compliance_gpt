from operator import itemgetter
from bson import ObjectId
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from database import rekam_jejak_vector, ketentuan_terkait_vector
from azure_config import azure_llm
from routing import router_chain
from constants import RAG_FUSION_PROMPT

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
        chain = {"question": itemgetter("question"),
                 "history" : itemgetter("history")} | retrieval_rekam_jejak_chain_rag_fusion
        return chain
    elif result["result"] == "ketentuan_terkait":
        chain = {"question": itemgetter("question"),
                 "history" : itemgetter("history")} | retrieval_ketentuan_terkait_chain_rag_fusion
        return chain

def rag_fusion_chain():
    context_chain = {
    "result":router,
    "question": itemgetter("question"),
    "history" : itemgetter("history")
    } | RunnablePassthrough() | {"context": RunnableLambda(choose_retriever), "question": itemgetter("question"), "history" : itemgetter("history") }

    return context_chain

llm = azure_llm()
router = router_chain()

template = RAG_FUSION_PROMPT
prompt_rag_fusion = ChatPromptTemplate.from_template(template)
generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

rekam_jejak_retriever = rekam_jejak_vector().as_retriever()
ketentuan_terkait_retriever = ketentuan_terkait_vector().as_retriever()

retrieval_rekam_jejak_chain_rag_fusion = generate_queries | rekam_jejak_retriever.map() | reciprocal_rank_fusion
retrieval_ketentuan_terkait_chain_rag_fusion = generate_queries | ketentuan_terkait_retriever.map() | reciprocal_rank_fusion

