import streamlit as st
import os 
import dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pymongo import MongoClient
from rag import caller

import time
import random
import string
# Initialize environment variables
dotenv.load_dotenv()

# sess_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
sess_id = "W56PNA34XM"

# Streamlit page configuration
st.set_page_config(page_title="Chat Application", layout="wide")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Your message here", key="user_input"):

    # Add our input to the session state
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    # Add our input to the chat window
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.spinner("Thinking ..."):
        response = caller(user_prompt, sess_id)
    # Add the response to the session state
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    # Add the response to the chat window
    with st.chat_message("assistant"):
        st.markdown(response)




