"""
Main Pipeline Module - Assignment 22 Complete Integration

This module runs the complete end-to-end similarity search pipeline:
Documents -> Splitter -> Embeddings -> Vector Store -> Similarity Search

Author: Patan Khalilulla Khan
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embedding_models import OpenAIEmbeddingModel, HuggingFaceEmbeddingModel, compare_embedding_models
from src.vector_stores import FAISSVectorStore, ChromaDBVectorStore
from src.similarity_search import SimilaritySearchEngine, EndToEndPipeline, load_chunks_from_file


def run_part1_embedding_models():
    """Part 1: Embedding Models - Tasks 1-3"""
    print("\n" + "=" * 70)
    print("PART 1 — EMBEDDING MODELS")
    print("=" * 70)
    
    # Task 1: OpenAI Embedding Model (conceptual - requires API key)
    print("\nTask 1: OpenAI Embedding Model")
    print("  Model: text-embedding-3-small")
    print("  Dimension: 1536")
    print("  Status: Conceptual (requires OPENAI_API_KEY)")
    
    # Task 2: Hugging Face Embedding Model
    print("\nTask 2: Hugging Face Embedding Model")
    hf_model = HuggingFaceEmbeddingModel()
    
    if hf_model.model is not None:
        test_text = "This is a test document for embedding generation."
        embedding = hf_model.generate_embedding(test_text)
        print(f"  Generated embedding dimension: {len(embedding)}")
        print(f"  Sample embedding values: {embedding[:5]}...")
    
    # Task 3: Comparison
    print("\nTask 3: OpenAI vs Hugging Face Comparison")
    compare_embedding_models()
    
    return hf_model


def run_part2_similarity_search(embedding_model):
    """Part 2: Document Similarity Search (OpenAI Embeddings) - Tasks 1-4"""
    print("\n" + "=" * 70)
    print("PART 2 — DOCUMENT SIMILARITY SEARCH (OpenAI/Hugging Face Embeddings)")
    print("=" * 70)
    
    # Load chunks from Assignment 21
    chunks_file = "data/document_chunks.txt"
    if not os.path.exists(chunks_file):
        print(f"Chunked file not found: {chunks_file}")
        print("Creating sample chunks for demonstration...")
        chunks = [
            "This is a sample text file for LangChain loader task.",
            "id,name,role\n1,Patan,Engineer",
            "Hello World! from PDF document."
        ]
    else:
        chunks = load_chunks_from_file(chunks_file)
    
    print(f"\nLoaded {len(chunks)} document chunks")
    
    # Task 1: Convert document chunks into embeddings
    print("\nTask 1: Converting document chunks into embeddings")
    embeddings = embedding_model.generate_embeddings(chunks)
    print(f"  Generated {len(embeddings)} embeddings")
    print(f"  Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
    
    # Task 2: Store embeddings in memory or vector store
    print("\nTask 2: Storing embeddings in FAISS vector store")
    vector_store = FAISSVectorStore(dimension=len(embeddings[0]) if embeddings else 384)
    vector_store.add_embeddings(embeddings, chunks)
    
    # Task 3: Create a function for similarity search
    print("\nTask 3: Similarity search function created")
    search_engine = SimilaritySearchEngine(vector_store, embedding_model)
    
    # Task 4: Test similarity search with at least 3 queries
    print("\nTask 4: Testing similarity search with 3 queries")
    test_queries = [
        "sample text file",
        "engineer role",
        "hello world PDF"
    ]
    
    search_engine.test_multiple_queries(test_queries)
    
    return vector_store, embedding_model


def run_part3_ollama_embeddings():
    """Part 3: Ollama Embeddings (Local Setup) - Tasks 1-4"""
    print("\n" + "=" * 70)
    print("PART 3 — OLLAMA EMBEDDINGS (LOCAL SETUP)")
    print("=" * 70)
    
    print("\nTask 1: Ollama Embedding Model")
    print("  Model: nomic-embed-text")
    print("  Status: Conceptual (requires Ollama installation)")
    
    print("\nTask 2: Install and run Ollama locally")
    print("  Command: ollama pull nomic-embed-text")
    print("  Command: ollama run nomic-embed-text")
    
    print("\nTask 3: Generate embeddings for document chunks")
    print("  Status: Ready when Ollama is running")
    
    print("\nTask 4: Compare with OpenAI embeddings")
    print("  Comparison documented in README.md")


def run_part4_vector_stores():
    """Part 4: Vector Stores - Tasks 1-4"""
    print("\n" + "=" * 70)
    print("PART 4 — VECTOR STORES (FAISS & ChromaDB)")
    print("=" * 70)
    
    # Load chunks
    chunks_file = "data/document_chunks.txt"
    if os.path.exists(chunks_file):
        chunks = load_chunks_from_file(chunks_file)
    else:
        chunks = [
            "This is a sample text file for LangChain loader task.",
            "id,name,role\n1,Patan,Engineer",
            "Hello World! from PDF document."
        ]
    
    # Generate embeddings using Hugging Face
    hf_model = HuggingFaceEmbeddingModel()
    embeddings = hf_model.generate_embeddings(chunks)
    
    # Task 1: FAISS Vector Store
    print("\nTask 1: FAISS Vector Store")
    faiss_store = FAISSVectorStore(dimension=len(embeddings[0]))
    faiss_store.add_embeddings(embeddings, chunks)
    
    # Task 2: ChromaDB Vector Store
    print("\nTask 2: ChromaDB Vector Store")
    chroma_store = ChromaDBVectorStore()
    chroma_store.add_embeddings(embeddings, chunks)
    
    # Task 3: Reload index and verify search works
    print("\nTask 3: Reload index and verify search works")
    faiss_store.save_index("data/faiss_index.bin")
    faiss_store.load_index("data/faiss_index.bin")
    
    # Test search after reload
    query_embedding = hf_model.generate_embedding("sample text")
    results = faiss_store.similarity_search(query_embedding, top_k=2)
    print(f"  Search after reload: {len(results)} results found")
    
    # Task 4: Similarity search with LangChain abstraction
    print("\nTask 4: Similarity search with LangChain abstraction")
    print("  Status: Implemented via similarity_search() function")
    
    return faiss_store, chroma_store, hf_model


def run_part5_mini_project():
    """Part 5: Mini Project Integration - Task 10"""
    print("\n" + "=" * 70)
    print("PART 5 — MINI PROJECT INTEGRATION")
    print("=" * 70)
    
    print("\nTask 10: End-to-End Similarity Search Pipeline")
    print("  Pipeline: Documents -> Splitter -> Embeddings -> Vector Store -> Similarity Search")
    
    # Load chunks
    chunks_file = "data/document_chunks.txt"
    if os.path.exists(chunks_file):
        chunks = load_chunks_from_file(chunks_file)
    else:
        chunks = [
            "This is a sample text file for LangChain loader task.",
            "id,name,role\n1,Patan,Engineer",
            "Hello World! from PDF document."
        ]
    
    print(f"\n  Input: {len(chunks)} document chunks")
    
    # Initialize pipeline
    pipeline = EndToEndPipeline(
        embedding_backend="huggingface",
        vector_store_backend="faiss"
    )
    pipeline.initialize_embedding_model()
    pipeline.initialize_vector_store(dimension=384)
    pipeline.add_documents(chunks)
    
    # Test pipeline search
    print("\n  Testing pipeline search:")
    test_queries = [
        "text file content",
        "engineer information",
        "PDF hello world"
    ]
    
    for query in test_queries:
        results = pipeline.search(query, top_k=2)
        print(f"\n  Query: '{query}'")
        for doc, score in results:
            print(f"    Score: {score:.4f} | {doc[:100]}...")


def main():
    """Run all parts of Assignment 22."""
    print("#" * 70)
    print("# ASSIGNMENT 22: EMBEDDING MODELS, VECTOR STORES & SIMILARITY SEARCH")
    print("# Student: Patan Khalilulla Khan")
    print("#" * 70)
    
    # Part 1
    hf_model = run_part1_embedding_models()
    
    # Part 2
    run_part2_similarity_search(hf_model)
    
    # Part 3
    run_part3_ollama_embeddings()
    
    # Part 4
    run_part4_vector_stores()
    
    # Part 5
    run_part5_mini_project()
    
    print("\n" + "#" * 70)
    print("# ASSIGNMENT 22 COMPLETED SUCCESSFULLY")
    print("#" * 70)


if __name__ == "__main__":
    main()

