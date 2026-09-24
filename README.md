# 📄 PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about PDF documents. The application retrieves relevant information from the document using semantic search and uses a Gemini LLM to generate grounded answers.

## 🚀 How It Works

```text
PDF Document
     ↓
PyPDFLoader
     ↓
Text Chunking
     ↓
HuggingFace Embeddings
     ↓
Chroma Vector Database
     ↓
Similarity Search
     ↓
Relevant Context
     ↓
Gemini LLM
     ↓
Answer
```

## 🛠️ Tech Stack

* Python
* LangChain
* Gemini API
* ChromaDB
* HuggingFace Embeddings
* PyPDFLoader
* Sentence Transformers

## ✨ Features

* PDF document loading
* Document chunking with overlap
* Semantic vector search
* HuggingFace embeddings
* Chroma vector database
* Context-aware question answering
* Source document retrieval
* Grounded responses using retrieved PDF context

## 📂 Project Structure

```text
Pdf-rag-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .
```
