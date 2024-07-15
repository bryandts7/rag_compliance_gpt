from operator import itemgetter
from bson import ObjectId
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain.memory.summary import SummarizerMixin
from langchain.retrievers import EnsembleRetriever

from database.database import rekam_jejak_vector, ketentuan_terkait_vector
from utils.azure_openai import azure_llm
from handler.routing import router_chain
from constants.prompt import RAG_FUSION_PROMPT, RAG_REKAM_JEJAK_PROMPT, SUMMARY_HISTORY_PROMPT
from database.graph_rag import graph_rag_chain
from handler.bm25 import bm25_ketentuan_retriever, bm25_rekam_retriever

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
    print("Retrieved Documents:", len(reranked_results))
    print(reranked_results)

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results[:8]

def history_summarize(history):
    global history_text
    sum_hist = history_sum.predict_new_summary(history, history_text)
    history_text = sum_hist
    print("History:", sum_hist)
    return sum_hist

def ketentuan_terkait_runnable(queries):
    res = [ketentuan_terkait_retriever_bm25.invoke(queries[0]), ketentuan_terkait_retriever.invoke(queries[1]), ketentuan_terkait_retriever.invoke(queries[2])]
    return res

def choose_retriever(result):
    print("Retriever routed to:", result["result"])
    if result["result"] == "rekam_jejak":
        parallel = RunnableParallel(unstructured=retrieval_rekam_jejak_chain_rag_fusion, structured=graph_chain)
        chain = {"question": itemgetter("question"), "query": itemgetter("question"),
                 "history" : itemgetter("history") | RunnableLambda(history_summarize)} | parallel | context_rekam_jejak
        return chain
    elif result["result"] == "ketentuan_terkait":
        chain = {"question": itemgetter("question"),
                 "history" : itemgetter("history") | RunnableLambda(history_summarize)} | retrieval_ketentuan_terkait_chain_rag_fusion
        return chain

def rag_fusion_chain():
    context_chain = {
    "result":router,
    "question": itemgetter("question"),
    "history" : itemgetter("history")
    } | RunnablePassthrough() | {"context": RunnableLambda(choose_retriever), "question": itemgetter("question"), "history" : itemgetter("history") }

    return context_chain

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"], template=SUMMARY_HISTORY_PROMPT
)


llm = azure_llm()

router = router_chain()

history_sum = SummarizerMixin(llm=llm, prompt=SUMMARY_PROMPT)
history_text = "-"

template = RAG_FUSION_PROMPT
prompt_rag_fusion = ChatPromptTemplate.from_template(template)
generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

rekam_jejak_retriever = rekam_jejak_vector().as_retriever(search_type="mmr")
ketentuan_terkait_retriever = ketentuan_terkait_vector().as_retriever(search_type="mmr")

rekam_jejak_retriever_bm25 = bm25_rekam_retriever()
rekam_jejak_retriever_bm25.k = 10

ketentuan_terkait_retriever_bm25 = bm25_ketentuan_retriever()
ketentuan_terkait_retriever_bm25.k = 10

# retrieval_rekam_jejak_chain_rag_fusion = generate_queries | rekam_jejak_retriever.map() | reciprocal_rank_fusion
# retrieval_ketentuan_terkait_chain_rag_fusion = generate_queries | ketentuan_terkait_retriever.map() | reciprocal_rank_fusion

retrieval_rekam_jejak_chain_rag_fusion = generate_queries | rekam_jejak_retriever.map() | reciprocal_rank_fusion
retrieval_ketentuan_terkait_chain_rag_fusion = generate_queries | RunnableLambda(ketentuan_terkait_runnable) | reciprocal_rank_fusion


graph_chain = graph_rag_chain()
template_rekam_jejak = RAG_REKAM_JEJAK_PROMPT
context_rekam_jejak = PromptTemplate(
    input_variables=["unstructured", "structured"], template=template_rekam_jejak
)

