import os
from dotenv import load_dotenv
import streamlit as st

# 1. LOAD ENV FIRST
load_dotenv()

# 2. (Optional but recommended) Verify it's actually loading
if not os.getenv("LANGCHAIN_API_KEY"):
    print("Warning: LANGCHAIN_API_KEY not found in environment!")

    
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"

# 2. SET TRACING VARS (Do this before importing LangChain members if possible)
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Project_1"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# 3. NOW IMPORT LANGCHAIN
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 4. INITIALIZE LLM 
# Note: Verified Groq ID is 'gpt-oss-120b' or 'llama-3.3-70b-versatile'
llm = ChatGroq(model='llama-3.3-70b-versatile') 
output_parser = StrOutputParser()

## Prompt Template 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries"),
    ("user", "Question:{question}")
])

## Chain
chain = prompt | llm | output_parser

## Streamlit UI
st.title('LangChain Demo with GROQ_API')
input_text = st.text_input("Search the topic you want")

# 5. FORCE INITIALIZATION TRACE
if 'initialized' not in st.session_state:
    try:
        # This creates the project in LangSmith immediately
        chain.invoke({"question": "Initial connection test"})
        st.session_state['initialized'] = True
        print("🚀 LangSmith Trace Sent Successfully!")
    except Exception as e:
        print(f"⚠️ Tracing failed: {e}")

if input_text:
    response = chain.invoke({'question': input_text})
    st.write(response)