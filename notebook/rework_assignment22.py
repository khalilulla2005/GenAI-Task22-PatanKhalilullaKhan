import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown: Title
text_cell_1 = nbf.v4.new_markdown_cell("""# Assignment 22: LangChain Embeddings & Vector Stores (Native Mastery)
**Student Name:** Patan Khalilulla Khan  
**Course:** GenAI Training Program  

---
## Assignment Overview
This notebook implements the core requirements for Assignment 22 using **native LangChain abstractions**:
1. **OpenAI Embeddings (Task 1):** Actual embedding generation and vector inspection.
2. **Native FAISS Integration (Task 5):** Using `langchain_community.vectorstores.FAISS` for creation and search.
3. **Local Ollama Integration (Task 6):** Using local embeddings via Ollama.
4. **Persistent ChromaDB (Task 8):** Persisting data to disk using `persist_directory`.
""")

# Code: Imports
code_cell_1 = nbf.v4.new_code_cell("""import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# Set API Key (Note: Ensure environment variable is set in your environment)
# os.environ["OPENAI_API_KEY"] = "sk-..."

print("Native LangChain modules imported successfully.")
""")

# Markdown: Task 1
text_cell_2 = nbf.v4.new_markdown_cell("""## Task 1: Generate OpenAI Embeddings & Inspect Vectors
Generating real embeddings for sample text and inspecting the resulting numerical vectors.
""")

# Code: Task 1
code_cell_2 = nbf.v4.new_code_cell("""try:
    embeddings_model = OpenAIEmbeddings()
    sample_text = "LangChain is a framework for developing applications powered by language models."
    
    # Generate actual embedding
    vector = embeddings_model.embed_query(sample_text)
    
    print(f"Embedding length: {len(vector)}")
    print(f"Vector preview (first 10 values): {vector[:10]}")
except Exception as e:
    print("OpenAI Embedding generation notice (API key needed):", e)
""")

# Markdown: Task 5
text_cell_3 = nbf.v4.new_markdown_cell("""## Task 5: Native FAISS Integration
Using LangChain's built-in FAISS abstraction to create a vector store and perform similarity search.
""")

# Code: Task 5
code_cell_3 = nbf.v4.new_code_cell("""texts = [
    "OpenAI embeddings are widely used for semantic search.",
    "FAISS is an efficient library for vector similarity search.",
    "ChromaDB provides persistent vector storage.",
    "Retrieval-Augmented Generation (RAG) grounds LLMs with external data."
]
docs = [Document(page_content=t) for t in texts]

try:
    # Task 5: Use FAISS.from_documents and similarity_search
    faiss_db = FAISS.from_documents(docs, embeddings_model)
    query = "How to perform vector search?"
    results = faiss_db.similarity_search(query, k=2)
    
    print("=== Task 5: FAISS Search Results ===")
    for doc in results:
        print(f"- {doc.page_content}")
except Exception as e:
    print("FAISS Integration notice:", e)
""")

# Markdown: Task 6
text_cell_4 = nbf.v4.new_markdown_cell("""## Task 6: Local Ollama Embedding Integration
Demonstrating integration with a local Ollama instance for embeddings (e.g., using `nomic-embed-text`).
""")

# Code: Task 6
code_cell_4 = nbf.v4.new_code_cell("""try:
    # Ensure Ollama is running and model is pulled: `ollama pull nomic-embed-text`
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Test local embedding
    local_vector = ollama_embeddings.embed_query("This is a test for local embeddings.")
    print(f"Local Embedding (Ollama) length: {len(local_vector)}")
except Exception as e:
    print("Local Ollama Embedding notice (ensure Ollama is running locally):", e)
""")

# Markdown: Task 8
text_cell_5 = nbf.v4.new_markdown_cell("""## Task 8: Persisting ChromaDB to Disk
Using `persist_directory` to ensure that our vector database is saved locally and can be reloaded.
""")

# Code: Task 8
code_cell_5 = nbf.v4.new_code_cell("""persist_dir = "chroma_db"

try:
    # Initialize Chroma with persistence
    chroma_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings_model,
        persist_directory=persist_dir
    )
    
    print(f"ChromaDB created and persisted to: {persist_dir}")
    
    # Verify search after persistence setup
    results = chroma_db.similarity_search("Tell me about ChromaDB", k=1)
    print(f"Verification result: {results[0].page_content}")
    
except Exception as e:
    print("ChromaDB Persistence notice:", e)
""")

nb['cells'] = [
    text_cell_1, code_cell_1,
    text_cell_2, code_cell_2,
    text_cell_3, code_cell_3,
    text_cell_4, code_cell_4,
    text_cell_5, code_cell_5
]

nbf.write(nb, 'D:\\techtude\\GenAI-Task22-PatanKhalilullaKhan\\notebook\\GenAI_Assignment22_Mastery_Pipeline.ipynb')
print("Assignment 22 Reworked Notebook (Mastery Edition) generated!")
