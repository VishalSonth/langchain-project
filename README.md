# 🤖 PDF-Based RAG Assistant with LangChain & Mistral-7B

This project is a **Retrieval-Augmented Generation (RAG)** pipeline that allows you to chat with your PDF documents. It uses **LangChain** for orchestration, **Hugging Face** for embeddings, and **Mistral-7B** for high-performance natural language generation.

## 🌟 Key Features
- **PDF Ingestion**: Automatically loads and parses complex PDF documents.
- **Smart Chunking**: Breaks down large documents into manageable segments for better context retrieval.
- **Semantic Search**: Uses vector embeddings to find exact information based on meaning, not just keywords.
- **Grounded Responses**: The AI only answers based on the provided PDF context, reducing hallucinations.

## 🛠️ Technology Stack
- **Framework**: [LangChain](https://www.langchain.com/)
- **LLM**: Mistral-7B-Instruct-v0.3 (via Hugging Face Endpoint)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
- **Embeddings**: HuggingFace BGE Embeddings
- **Environment**: Python 3.10+ & Jupyter Notebook

## 🚀 How It Works
1. **Load**: `PyPDFLoader` reads the raw text from your PDF files.
2. **Split**: `RecursiveCharacterTextSplitter` cuts the text into 1000-character chunks with a 200-character overlap to preserve context.
3. **Embed**: `HuggingFaceBgeEmbeddings` converts text chunks into mathematical vectors.
4. **Store**: Vectors are stored in a **FAISS** index for lightning-fast retrieval.
5. **Retrieve & Answer**: When a query is made, the system finds the most relevant chunks and passes them to **Mistral-7B** to generate a precise answer.

## ⚙️ Setup & Installation

### 1. Clone the repository
bash
git clone [https://github.com/VishalSonth/langchain-project.git](https://github.com/VishalSonth/langchain-project.git)
cd langchain-project
2. Install dependencies
Bash
pip install -r requirements.txt
3. Configure Environment Variables
Create a .env file in the root directory (the project includes a .env.example as a template):

Plaintext
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
4. Run the Project
Open the Jupyter Notebook and run the cells in order:

Bash
jupyter notebook huggingface/huggingface.ipynb
📂 Project Structure
huggingface/: Contains the main RAG notebook using Hugging Face models.

rag/: Additional RAG implementation experiments.

agents/: Experiments with LangChain agents.

data/: (Optional) Directory for your input PDF files.
