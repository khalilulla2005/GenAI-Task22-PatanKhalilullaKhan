"""
Embedding Models Module - Part 1 of Assignment 22

This module implements embedding generation using:
- OpenAI Embedding Model (Task 1)
- Hugging Face Embedding Model (Task 2)
- Comparison between both (Task 3)

Author: Patan Khalilulla Khan
"""

from typing import List, Dict, Any
import os


class OpenAIEmbeddingModel:
    """
    Task 1: OpenAI Embedding Model
    
    Uses OpenAI's text-embedding-3-small model (or latest available)
    to convert text into dense vector embeddings.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client if API key is available."""
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                print(f"OpenAI Embedding Model initialized: {self.model_name}")
            else:
                print("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        except ImportError:
            print("OpenAI package not installed. Install with: pip install openai")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if self.client is None:
            return [0.0] * 1536  # Return zero vector as fallback
        
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        return [self.generate_embedding(text) for text in texts]
    
    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension for this model."""
        return 1536  # text-embedding-3-small default


class HuggingFaceEmbeddingModel:
    """
    Task 2: Hugging Face Embedding Model
    
    Uses sentence-transformers/all-MiniLM-L6-v2 model
    to convert text into dense vector embeddings locally.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Hugging Face sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"Hugging Face Embedding Model initialized: {self.model_name}")
        except ImportError:
            print("sentence-transformers not installed. Install with: pip install sentence-transformers")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if self.model is None:
            return [0.0] * 384  # Return zero vector as fallback
        
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            return [[0.0] * 384 for _ in texts]
        
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension for this model."""
        return 384  # all-MiniLM-L6-v2 default


def compare_embedding_models() -> Dict[str, Any]:
    """
    Task 3: OpenAI vs Hugging Face Embeddings Comparison
    
    Returns comparison results including:
    - Model names
    - Embedding dimensions
    - Performance characteristics
    - Use case recommendations
    """
    comparison = {
        "OpenAI Embedding Model": {
            "model": "text-embedding-3-small",
            "dimension": 1536,
            "provider": "OpenAI (API)",
            "pros": [
                "State-of-the-art performance",
                "High accuracy for semantic similarity",
                "Easy API integration",
                "Regularly updated by OpenAI"
            ],
            "cons": [
                "Requires API key and payment",
                "Rate limits apply",
                "Data sent to external service",
                "Internet dependency"
            ],
            "best_for": "Production applications requiring highest accuracy"
        },
        "Hugging Face Embedding Model": {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
            "provider": "Hugging Face (Local)",
            "pros": [
                "Runs locally (no API cost)",
                "No internet required after download",
                "Data privacy (stays on local machine)",
                "Fast inference on CPU/GPU",
                "Open source"
            ],
            "cons": [
                "May have slightly lower accuracy than OpenAI",
                "Requires local compute resources",
                "Model download needed initially",
                "Manual updates required"
            ],
            "best_for": "Development, privacy-sensitive, cost-conscious applications"
        }
    }
    
    print("=" * 70)
    print("EMBEDDING MODEL COMPARISON: OpenAI vs Hugging Face")
    print("=" * 70)
    
    for model_name, details in comparison.items():
        print(f"\n{model_name}")
        print(f"  Model: {details['model']}")
        print(f"  Dimension: {details['dimension']}")
        print(f"  Provider: {details['provider']}")
        print(f"  Best for: {details['best_for']}")
        print(f"  Pros: {', '.join(details['pros'])}")
        print(f"  Cons: {', '.join(details['cons'])}")
    
    return comparison
