import streamlit as st 
import os 
import time 
from dotenv import load_dotenv

# --- Modern 2026 Core & Partner Imports ---
from langchain_groq import ChatGroq 
from langchain_ollama import OllamaEmbeddings 
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configuration & Env Setup
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# 2. Vector Store Initialization (Cached for performance)
if "vector" not in st.session_state:
    with st.spinner("Initializing Vector Database..."):
        # We use a smaller, faster embedding model for 2026 local RAG
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# 3. UI Setup
st.title("🚀 ChatGroq RAG Demo (2026 Edition)")

# Use Llama 3 on Groq for blazing fast responses
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# 4. Prompt Definition
qa_template = """ 
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
""" 
prompt = ChatPromptTemplate.from_template(qa_template)

# 5. The Modern LCEL Chain (Replaces create_retrieval_chain)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = st.session_state.vectors.as_retriever()

# This is the "Pipe" architecture: Retrieval -> Prompt -> LLM -> String
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. User Interaction
user_input = st.text_input("What would you like to know about LangSmith?")

if user_input:
    start_time = time.process_time()
    
    # In 2026, we invoke the chain directly
    response = rag_chain.invoke(user_input)
    
    st.info(f"Response time: {time.process_time() - start_time:.2f} seconds")
    st.markdown(f"### Answer:\n{response}")

    # 7. Document Similarity Search (Debugging Tool)
    with st.expander("Document Similarity Search (Context Chunks)"):
        # We manually retrieve relevant docs to show the user what the AI "read"
        relevant_docs = retriever.invoke(user_input)
        for i, doc in enumerate(relevant_docs):
            st.write(f"**Chunk {i+1}:**")
            st.write(doc.page_content)
            st.write("---")