"""
Vector Stores Module - Part 4 of Assignment 22

This module implements vector store functionality using:
- FAISS Vector Store (Task 1)
- ChromaDB Vector Store (Task 2)
- Similarity search and index management (Tasks 3-4)

Author: Patan Khalilulla Khan
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple


class FAISSVectorStore:
    """
    Task 1: FAISS Vector Store
    
    Facebook AI Similarity Search (FAISS) is a library for efficient
    similarity search and clustering of dense vectors.
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        self.documents = []
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        try:
            import faiss
            self.index = faiss.IndexFlatL2(self.dimension)
            print(f"FAISS Vector Store initialized (dimension: {self.dimension})")
        except ImportError:
            print("faiss-cpu not installed. Install with: pip install faiss-cpu")
    
    def add_embeddings(self, embeddings: List[List[float]], documents: List[str]):
        """
        Store document embeddings in FAISS index.
        
        Args:
            embeddings: List of embedding vectors
            documents: List of corresponding document texts
        """
        if self.index is None:
            print("FAISS index not initialized")
            return
        
        import faiss
        vectors = np.array(embeddings).astype('float32')
        self.index.add(vectors)
        self.documents.extend(documents)
        print(f"FAISS: Added {len(embeddings)} embeddings. Total: {len(self.documents)}")
    
    def similarity_search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Task 4: Test similarity search with at least 3 queries.
        
        Find most similar documents to the query embedding.
        
        Args:
            query_embedding: Query vector
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        import faiss
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, min(top_k, len(self.documents)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                similarity_score = 1 / (1 + distances[0][i])  # Convert distance to similarity
                results.append((self.documents[idx], similarity_score))
        
        return results
    
    def save_index(self, filepath: str):
        """Task 3: Reload index and verify search works."""
        if self.index is None:
            return
        import faiss
        faiss.write_index(self.index, filepath)
        # Save documents separately
        with open(filepath + '.docs', 'w', encoding='utf-8') as f:
            json.dump(self.documents, f)
        print(f"FAISS index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Reload saved index."""
        if self.index is None:
            self._initialize_index()
        import faiss
        self.index = faiss.read_index(filepath)
        with open(filepath + '.docs', 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        print(f"FAISS index loaded from {filepath}. Documents: {len(self.documents)}")


class ChromaDBVectorStore:
    """
    Task 2: ChromaDB Vector Store
    
    ChromaDB is an open-source embedding database for building
    AI applications with embeddings.
    """
    
    def __init__(self, collection_name: str = "document_chunks"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(name=self.collection_name)
            print(f"ChromaDB Vector Store initialized (collection: {self.collection_name})")
        except ImportError:
            print("chromadb not installed. Install with: pip install chromadb")
    
    def add_embeddings(self, embeddings: List[List[float]], documents: List[str], ids: List[str] = None):
        """
        Store document embeddings in ChromaDB.
        
        Args:
            embeddings: List of embedding vectors
            documents: List of corresponding document texts
            ids: Optional list of unique IDs for each document
        """
        if self.collection is None:
            print("ChromaDB collection not initialized")
            return
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
        print(f"ChromaDB: Added {len(embeddings)} embeddings. Total: {self.collection.count()}")
    
    def similarity_search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find most similar documents to the query embedding.
        
        Args:
            query_embedding: Query vector
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.collection is None:
            return []
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )
        
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        # Convert distance to similarity score
        results_list = []
        for doc, dist in zip(documents, distances):
            similarity_score = 1 / (1 + dist)
            results_list.append((doc, similarity_score))
        
        return results_list
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection."""
        if self.collection is None:
            return {}
        return {
            "name": self.collection_name,
            "count": self.collection.count()
        }
