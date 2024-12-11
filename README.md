# RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot built using LangChain, AWS Bedrock, and Chroma.

## Features
- Supports Claude LLM through AWS Bedrock
- Vector-based retrieval with Chroma
- Dynamically adds new documents for RAG updates

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/RAG-Chatbot.git

2. install dependencies
    pip install -r requirments.txt

## Usage
1. LLM Integration (Bedrock + Claude Model)
    ```bash
    streamlit run frontend.py

2. Vector Store (RAG) Construction
    ```bash
    streamlit run frontend_1.py

### How to Update the Knowledge Base

The chatbot relies on a knowledge base stored locally in the docs/ directory. To update or expand the knowledge base, you can add .txt files containing the desired information. These files will be processed and embedded into the vector store for retrieval during queries.