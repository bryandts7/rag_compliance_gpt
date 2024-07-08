from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph

from azure_config import azure_llm
from constants import GRAPH_QA_GEN_PROMPT

import dotenv
import os
dotenv.load_dotenv()

URL = os.getenv("NEO4J_GRAPH_URL")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE")

graph = Neo4jGraph(url=URL, username=USERNAME, password= PASSWORD, database=DATABASE, )
llm = azure_llm()

qa_generation_template = GRAPH_QA_GEN_PROMPT
qa_generation_prompt = PromptTemplate(input_variables=["context", "question"], template=qa_generation_template)

def graph_rag_chain():
    chain = GraphCypherQAChain.from_llm(
    cypher_llm = llm, qa_llm = llm, graph=graph, verbose=True, qa_prompt=qa_generation_prompt, validate_cypher=True,
    )
    return chain