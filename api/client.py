import requests 
import streamlit as st 

def get_chatGroq_response(input_text):
    try:
        # Changed to 127.0.0.1 for better Windows compatibility
        response = requests.post(
            "http://127.0.0.1:8000/essay/invoke",
            json={'input': {'topic': input_text}}
        )
        # LangServe ChatModels return a dict with 'content'
        return response.json()['output']['content']
    except Exception as e:
        return f"Error connecting to Groq server: {e}"

def get_Ollama_response(input_text):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/poem/invoke",
            json={'input': {'topic': input_text}}
        )
        # Plain LLMs (Ollama) might return just the string in 'output'
        # Let's handle both cases
        res_json = response.json()
        output = res_json['output']
        return output if isinstance(output, str) else output.get('content', output)
    except Exception as e:
        return f"Error connecting to Ollama server: {e}"

## Streamlit Framework 
st.title('Langchain Demo with API')
input_text = st.text_input("Write an essay on")
input_text1 = st.text_input("Write a Poem on")

if input_text:
    st.write(get_chatGroq_response(input_text))

if input_text1:
    st.write(get_Ollama_response(input_text1))



