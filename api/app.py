from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes
import uvicorn
import os 
from langchain_community.llms import Ollama
#from langchain_Ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"

)


# Initialize Models

model=ChatGroq(model="llama-3.3-70b-versatile")
## Ollama llama2
llm=Ollama(model="llama2")
#llm=OllamaLLM(model="llama2")


## Prompt Templates
prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 100 words")


# Adding routes 

add_routes(
    app,
    model,
    path="/chatGroq"
)

add_routes(
    app,
    prompt1 | model,
    path="/essay"
)


add_routes(
    app,
    prompt2 | llm,
    path="/poem"
)

if __name__== "__main__":
    uvicorn.run(app, host="localhost", port=8000)
