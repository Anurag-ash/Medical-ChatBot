# README: Medical ChatBot
Developed an AI-powered medical chatbot using Retrieval-Augmented Generation (RAG) to provide accurate, evidence-based answers from the Encyclopedia of Medical Science. Processed raw PDFs into a FAISS vector database using Hugging Face embeddings for efficient document retrieval and integrated Mistral-7B-Instruct via LangChain’s RetrievalQA with a custom prompt template to ensure clinically relevant responses. Designed an intuitive Streamlit-based UI for seamless interaction, enabling users to query complex medical topics with reliable, context-aware answers. 
## Prerequisite: Install Pipenv
Follow the official Pipenv installation guide to set up Pipenv on your system:  
[Install Pipenv Documentation](https://pipenv.pypa.io/en/latest/installation.html)

---

## Steps to Set Up the Environment

### Install Required Packages
Run the following commands in your terminal (assuming Pipenv is already installed):

```bash
pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf
pipenv install huggingface_hub
pipenv install streamlit



