# GenAI-Task22-PatanKhalilullaKhan

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-VectorStore-orange.svg)](https://github.com/facebookresearch/faiss)

> **Assignment 22:** Embedding Models, Vector Stores & Similarity Search for Retrieval-Augmented Generation (RAG).

---

## 🚀 Overview

This repository contains the professional implementation of **Assignment 22**, focusing on the foundational components of **Retrieval-Augmented Generation (RAG)** systems:
1. **Embedding Models:** Translating unstructured text into dense vector spaces (OpenAI & Hugging Face).
2. **Vector Stores:** Efficient indexing and retrieval using **FAISS** and **ChromaDB**.
3. **Similarity Search:** Semantic querying pipelines utilizing cosine/L2 distance metrics.

---

## 📂 Project Structure

```text
GenAI-Task22-PatanKhalilullaKhan/
├── data/                      # Input datasets and chunked documents
│   ├── sample.txt             # Plain text sample
│   ├── sample.csv             # Tabular sample
│   ├── sample.pdf             # PDF sample
│   └── document_chunks.txt    # Processed chunks (from Assignment 21)
├── src/                       # Modular source code
│   ├── __init__.py            # Package initialization
│   ├── embedding_models.py    # OpenAI & Hugging Face embedding engines
│   ├── vector_stores.py       # FAISS & ChromaDB wrappers
│   └── similarity_search.py   # Search engine & end-to-end pipeline
├── main_pipeline.py           # Master execution script (All Parts)
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT Open Source License
└── README.md                  # Comprehensive technical documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/khalilulla2005/GenAI-Task22-PatanKhalilullaKhan.git
   cd GenAI-Task22-PatanKhalilullaKhan
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete pipeline:**
   ```bash
   python main_pipeline.py
   ```

---

## 📋 Implementation Summary

| Part | Focus Area | Technologies | Status |
| :--- | :--- | :--- | :--- |
| **Part 1** | Embedding Models | OpenAI `text-embedding-3-small`, HF `all-MiniLM-L6-v2` | ✅ Implemented & Tested |
| **Part 2** | Document Similarity Search | NumPy, Cosine/L2 Distance, Custom Search Engine | ✅ Implemented & Tested |
| **Part 3** | Ollama Embeddings | Local `nomic-embed-text` setup | ✅ Conceptual / Ready |
| **Part 4** | Vector Stores | FAISS IndexFlatL2, ChromaDB Persistence | ✅ Implemented & Tested |
| **Part 5** | Mini-Project Integration | End-to-End RAG Retrieval Pipeline | ✅ Implemented & Tested |

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---
**Author:** Patan Khalilulla Khan  
**Course:** GenAI Training Program
