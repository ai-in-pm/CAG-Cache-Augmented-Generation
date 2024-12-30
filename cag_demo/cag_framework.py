"""
Core implementation of the Cache-Augmented Generation (CAG) Framework.
"""

import logging
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

from cag_demo.config import CAG_CONFIG, LLM_CONFIGS
from cag_demo.llm_interface import LLMInterface

class CAGFramework:
    def __init__(self, llm_name: str, knowledge_base_path: Path):
        """
        Initialize the CAG Framework.
        
        Args:
            llm_name: Name of the LLM to use
            knowledge_base_path: Path to the knowledge base directory
        """
        self.logger = logging.getLogger(__name__)
        self.llm_config = LLM_CONFIGS[llm_name]
        self.cache = {}
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize LLM interface
        self.llm = LLMInterface(llm_name)
        
        # Initialize tokenizer and model for embedding generation
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        
        # Load and preprocess knowledge base
        self._preload_knowledge_base()

    def _preload_knowledge_base(self):
        """Preload and preprocess the knowledge base into the KV cache."""
        self.logger.info("Preloading knowledge base...")
        
        # Load documents from knowledge base
        documents = self._load_documents()
        
        # Process documents into chunks
        self.chunks = self._chunk_documents(documents)
        
        # Generate embeddings for chunks
        self.embeddings = self._generate_embeddings(self.chunks)
        
        self.logger.info(f"Preloaded {len(self.chunks)} key-value pairs into cache")

    def _load_documents(self) -> List[str]:
        """Load documents from the knowledge base directory."""
        documents = []
        for file_path in self.knowledge_base_path.glob("**/*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())
        return documents

    def _chunk_documents(self, documents: List[str]) -> List[str]:
        """Split documents into smaller chunks."""
        chunks = []
        chunk_size = CAG_CONFIG["preload_chunk_size"]
        
        for doc in documents:
            # Simple splitting by chunk size
            # In production, use more sophisticated chunking strategies
            doc_chunks = [doc[i:i + chunk_size] 
                         for i in range(0, len(doc), chunk_size)]
            chunks.extend(doc_chunks)
        
        return chunks

    def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks."""
        embeddings = []
        
        with torch.no_grad():
            for chunk in tqdm(chunks, desc="Generating embeddings"):
                inputs = self.tokenizer(chunk, return_tensors="pt", 
                                      max_length=512, truncation=True)
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(embedding)
        
        return np.vstack(embeddings)

    async def query(self, query: str) -> Tuple[str, Dict]:
        """
        Query the CAG framework with a question.
        
        Args:
            query: Question to ask
            
        Returns:
            Tuple containing:
                - Generated response
                - Dictionary of metrics
        """
        start_time = time.time()
        metrics = {'cache_hits': 0}
        
        # Generate query embedding
        query_embedding = self._generate_embeddings([query])[0]
        
        # Find most similar chunks
        similarities = np.dot(self.embeddings, query_embedding)
        top_k_indices = np.argsort(similarities)[-3:][::-1]  # Get top 3 most similar chunks
        
        # Construct prompt with relevant context
        context = "\n".join([self.chunks[i] for i in top_k_indices])
        prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate response
        try:
            response = await self.llm.generate_response(
                prompt=prompt,
                system_prompt="You are a helpful AI assistant that answers questions based on the provided context.",
                max_tokens=500  # Limit response length
            )
        except Exception as e:
            response = f"Error generating response: {str(e)}"
        
        metrics['response_time'] = time.time() - start_time
        metrics['memory_usage'] = len(self.chunks)
        
        return response, metrics

    def clear_cache(self):
        """Clear the KV cache."""
        self.cache.clear()
        self.logger.info("Cache cleared")
