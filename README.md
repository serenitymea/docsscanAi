# RAG System — Q&A on Any Text (FastAPI + VoyageAI + Gemini + ChromaDB)

A simple RAG (Retrieval-Augmented Generation) system that takes any text (article, document, product description, etc.)  
and answers questions about it, citing the source of the answer.  
It runs on **FastAPI**, **VoyageAI** (embeddings), **Gemini** (LLM), and **ChromaDB** (vector database).

<img width="1598" height="771" alt="image" src="https://github.com/user-attachments/assets/a8780014-52b2-4c15-b0cd-171557464b29" />

---

## How It Works

1. You send any text through the API.  
2. The text is automatically split into chunks, embedded using **VoyageAI**, and stored in **ChromaDB**.  
3. When a query is made, the system:
   - retrieves relevant fragments from ChromaDB,  
   - sends them to **Gemini**,  
   - receives a response, and returns it along with the sources (the text chunks the answer was derived from).

---

## Core Technologies

| Component | Purpose |
|------------|----------|
| **FastAPI** | HTTP API interface |
| **VoyageAI** | Embedding generation |
| **Gemini** | LLM for answer generation |
| **ChromaDB** | Vector database |
| **uvicorn** | Server runner |

---

## Installation
git clone https://github.com/serenitymea/docsscanAi
cd docsscanAi

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python web_i.py web --voyage-key .key here. --gemini-key .key here.


