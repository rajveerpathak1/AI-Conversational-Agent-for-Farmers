# AI Conversational Agent for Farmers

An AI-powered conversational assistant designed to provide agricultural guidance using Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and semantic document search.

The project focuses on building a scalable and modular AI system capable of answering agriculture-related queries using domain-specific knowledge bases such as crop guides, government schemes, irrigation manuals, and farming best practices.

---

# Features

* Conversational AI for agriculture-related queries
* Retrieval-Augmented Generation (RAG) pipeline
* Semantic search using FAISS vector database
* Multi-turn contextual conversations
* Open-source LLM integration (Llama 3)
* Groq API integration for fast inference
* Flask-based backend APIs
* Document ingestion and embedding pipeline
* Modular architecture for future voice and multilingual support

---

# Tech Stack

## Backend

* Python
* Flask

## AI / LLM

* LangChain
* Llama 3
* Groq API

## RAG Pipeline

* FAISS
* HuggingFace Embeddings
* Sentence Transformers

## Document Processing

* PyPDF
* Recursive Text Chunking

---

# System Architecture

```text
User Query
    ↓
Flask API
    ↓
LangChain Conversational Chain
    ↓
Retriever (FAISS Vector Store)
    ↓
Relevant Context Retrieval
    ↓
Llama 3 via Groq API
    ↓
AI Response
```

---

# Project Structure

```bash
AI-Conversational-Agent-for-Farmers/
│
├── app.py
├── requirements.txt
├── .env
│
├── chatbot/
│   └── chain.py
│
├── rag/
│   ├── ingest.py
│   └── retriever.py
│
├── data/
│   └── agriculture_docs/
│
├── vectorstore/
│
├── templates/
│   └── index.html
│
├── static/
│
└── README.md
```

---

# Installation & Setup

## 1. Clone Repository

```bash
git clone https://github.com/rajveerpathak1/AI-Conversational-Agent-for-Farmers.git

cd AI-Conversational-Agent-for-Farmers
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Setup Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Get API key from:

* [GroqCloud](https://console.groq.com)

---

# 5. Add Agricultural Documents

Place PDFs inside:

```bash
data/agriculture_docs/
```

Examples:

* Crop guides
* Government schemes
* Fertilizer manuals
* Irrigation documents

---

# 6. Create Vector Database

Run:

```bash
python rag/ingest.py
```

This will:

* Load PDFs
* Split documents into chunks
* Generate embeddings
* Store vectors in FAISS

---

# 7. Run Application

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

# RAG Pipeline

The project uses Retrieval-Augmented Generation (RAG) to improve factual accuracy and domain relevance.

## Workflow

1. Documents are loaded and chunked
2. Embeddings are generated using transformer models
3. FAISS stores vector embeddings
4. User queries are semantically matched
5. Retrieved context is passed to the LLM
6. Final response is generated

---

# Future Enhancements

* Voice-based interaction
* Indic language support
* Crop disease detection
* Weather integration
* Recommendation systems
* LangGraph-based agent workflows
* Multi-user session management
* Deployment with Docker & Kubernetes

---

# Deployment

The application can be deployed using:

---

# Sample Use Cases

* Crop recommendation assistance
* Government scheme guidance
* Fertilizer and irrigation advisory
* Farming best practices retrieval
* Agricultural knowledge support chatbot

---

# Learning Outcomes

Through this project, I explored:

* LLM orchestration using LangChain
* RAG architecture design
* Vector databases and semantic retrieval
* Conversational memory handling
* Backend API development
* Open-source AI ecosystem

---

# Author

**Rajveer Pathak**

* [LinkedIn](https://www.linkedin.com/in/rajveerpathak/)
* [GitHub](https://github.com/rajveerpathak1)

---

# License

This project is developed for learning, research, and educational purposes.
