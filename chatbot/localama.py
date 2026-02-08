from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st   
import os 
from dotenv import load_dotenv

load_dotenv()

# SET TRACING VARS (Do this before importing LangChain members if possible)

os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = "Project_1"
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

## Prompt Template 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries"),
    ("user", "Question:{question}")
])

## Streamlit UI
st.title('LangChain Demo with LLAMA2_API')
input_text = st.text_input("Search the topic you want")


# Ollama LLAma2 LLM
llm = Ollama(model="llama2") 
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    response = chain.invoke({'question': input_text})
    st.write(response)