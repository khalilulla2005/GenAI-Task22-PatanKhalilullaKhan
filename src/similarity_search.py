"""
Similarity Search Module - Part 2 & Part 5 of Assignment 22

This module implements:
- Document Similarity Search using OpenAI Embeddings (Part 2, Tasks 1-4)
- End-to-End Similarity Search Pipeline (Part 5, Task 10)

Author: Patan Khalilulla Khan
"""

from typing import List, Tuple, Dict, Any
from src.embedding_models import OpenAIEmbeddingModel, HuggingFaceEmbeddingModel
from src.vector_stores import FAISSVectorStore, ChromaDBVectorStore


class SimilaritySearchEngine:
    """
    Task 4: Similarity Search with at Least 3 Queries
    
    Performs similarity search on stored document embeddings.
    """
    
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Perform similarity search for a query string.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.generate_embedding(query)
        
        # Search in vector store
        results = self.vector_store.similarity_search(query_embedding, top_k=top_k)
        
        return results
    
    def test_multiple_queries(self, queries: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Task 4: Test similarity search with at least 3 queries.
        
        Args:
            queries: List of search queries
            
        Returns:
            Dictionary mapping query to search results
        """
        all_results = {}
        
        print("\n" + "=" * 70)
        print("SIMILARITY SEARCH TEST RESULTS")
        print("=" * 70)
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: '{query}'")
            print("-" * 70)
            
            results = self.search(query, top_k=3)
            all_results[query] = results
            
            for j, (doc, score) in enumerate(results, 1):
                print(f"  Result {j} (Score: {score:.4f}):")
                print(f"    {doc[:200]}...")
        
        print("\n" + "=" * 70)
        return all_results


class EndToEndPipeline:
    """
    Task 10: End-to-End Similarity Search Pipeline
    
    Complete pipeline: Documents → Splitter → Embeddings → Vector Store → Similarity Search
    """
    
    def __init__(self, embedding_backend: str = "huggingface", vector_store_backend: str = "faiss"):
        self.embedding_backend = embedding_backend
        self.vector_store_backend = vector_store_backend
        self.embedding_model = None
        self.vector_store = None
        self.search_engine = None
    
    def initialize_embedding_model(self):
        """Initialize the selected embedding backend."""
        if self.embedding_backend == "openai":
            self.embedding_model = OpenAIEmbeddingModel()
        elif self.embedding_backend == "huggingface":
            self.embedding_model = HuggingFaceEmbeddingModel()
        else:
            raise ValueError(f"Unknown embedding backend: {self.embedding_backend}")
        
        print(f"Embedding model initialized: {self.embedding_backend}")
    
    def initialize_vector_store(self, dimension: int):
        """Initialize the selected vector store backend."""
        if self.vector_store_backend == "faiss":
            self.vector_store = FAISSVectorStore(dimension=dimension)
        elif self.vector_store_backend == "chromadb":
            self.vector_store = ChromaDBVectorStore()
        else:
            raise ValueError(f"Unknown vector store backend: {self.vector_store_backend}")
        
        print(f"Vector store initialized: {self.vector_store_backend}")
    
    def add_documents(self, documents: List[str]):
        """
        Add documents to the pipeline (generate embeddings and store).
        
        Args:
            documents: List of document text chunks
        """
        # Generate embeddings
        embeddings = self.embedding_model.generate_embeddings(documents)
        
        # Add to vector store
        self.vector_store.add_embeddings(embeddings, documents)
        
        # Initialize search engine
        self.search_engine = SimilaritySearchEngine(self.vector_store, self.embedding_model)
        
        print(f"Pipeline: Added {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Perform similarity search through the complete pipeline.
        
        Args:
            query: Search query text
            top_k: Number of top results
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.search_engine is None:
            raise ValueError("Pipeline not initialized. Call add_documents() first.")
        
        return self.search_engine.search(query, top_k=top_k)


def load_chunks_from_file(filepath: str) -> List[str]:
    """
    Load document chunks from the chunked text file created in Assignment 21.
    
    Args:
        filepath: Path to document_chunks.txt
        
    Returns:
        List of chunk strings
    """
    chunks = []
    current_chunk = ""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('--- Chunk'):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = ""
            elif line:
                if current_chunk:
                    current_chunk += " " + line
                else:
                    current_chunk = line
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks
