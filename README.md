# chat_history
```
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.runnable import RunnableLambda

import os
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime

# Load environment variables
load_dotenv()
os.environ['HUGGING_FACE_API_TOKEN'] = os.getenv('HUGGING_FACE_API_TOKEN')
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

# Initialize embeddings and LLM
model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant")

# Setup Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "you are help ful assistant think step by step and give answer using you knowledge. address me like sir and start greet like jarvis in toney stark here is context {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
stuff_doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "db" not in st.session_state:
    st.session_state.db = None
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = None

# File uploader
st.title("📚 PDF Chat Assistant")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and not st.session_state.db:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader(uploaded_file.name)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    chunks = text_splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, model)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, stuff_doc_chain)
    retrieval_chain = retrieval_chain | RunnableLambda(lambda x: {"output": x["answer"], "context_data": x})

    st.session_state.db = db

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.chat_history:
            st.session_state.chat_history[session_id] = ChatMessageHistory()
        return st.session_state.chat_history[session_id]

    chat_history_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    st.session_state.chat_chain = chat_history_chain

# Chat interface
if st.session_state.chat_chain:
    config = {"configurable": {"session_id": "chat1"}}
    user_input = st.chat_input("Ask me anything")

    if user_input:
        with st.spinner("Jarvis is thinking..."):
            result = st.session_state.chat_chain.invoke({"input": user_input}, config=config)
            print(result)
            print()

    st.divider()
    chat_hist = st.session_state.chat_history.get("chat1")
    # print(type(chat_hist))
    if chat_hist:
        for msg in chat_hist.messages:
            role = "User" if msg.type == "human" else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)
```
