"""
Implementation of the Retrieval-Augmented Generation (RAG) Framework for comparison.
"""

import logging
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

from cag_demo.config import RAG_CONFIG, LLM_CONFIGS
from cag_demo.llm_interface import LLMInterface

class RAGFramework:
    def __init__(self, llm_name: str, documents_path: Path):
        """
        Initialize the RAG Framework.
        
        Args:
            llm_name: Name of the LLM to use
            documents_path: Path to the documents directory
        """
        self.logger = logging.getLogger(__name__)
        self.llm_config = LLM_CONFIGS[llm_name]
        self.documents_path = documents_path
        
        # Initialize LLM interface
        self.llm = LLMInterface(llm_name)
        
        # Initialize tokenizers and models
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        
        # Load and process documents
        self.documents = self._load_documents()
        self.document_embeddings = self._generate_embeddings(self.documents)
        
        # Initialize BM25 for sparse retrieval
        tokenized_docs = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def _load_documents(self) -> List[str]:
        """Load documents from the documents directory."""
        documents = []
        for file_path in self.documents_path.glob("**/*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())
        return documents

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate dense embeddings for texts."""
        embeddings = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Generating embeddings"):
                inputs = self.tokenizer(text, return_tensors="pt", 
                                      max_length=512, truncation=True)
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(embedding)
        
        return np.vstack(embeddings)

    async def query(self, query: str) -> Tuple[str, Dict]:
        """
        Process a query using the RAG Framework.
        
        Args:
            query: User query string
            
        Returns:
            tuple: (response, metrics)
        """
        start_time = time.time()
        metrics = {}
        
        # Generate query embedding
        query_embedding = self._generate_embeddings([query])[0]
        
        # Retrieve relevant documents
        documents = self._retrieve_documents(query_embedding)
        
        # Combine documents into context
        context = "\n".join(documents[:2])  # Limit to top 2 documents
        
        # Generate response
        prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            response = await self.llm.generate_response(
                prompt=prompt,
                system_prompt="You are a helpful AI assistant that answers questions based on the provided context.",
                max_tokens=500  # Limit response length
            )
        except Exception as e:
            response = f"Error generating response: {str(e)}"
        
        metrics['response_time'] = time.time() - start_time
        metrics['num_retrieved'] = len(documents)
        metrics['retriever_type'] = 'hybrid'
        
        return response, metrics

    def _retrieve_documents(self, query_embedding: np.ndarray) -> List[str]:
        """Retrieve relevant documents based on the query embedding."""
        # Calculate similarities
        similarities = np.dot(self.document_embeddings, query_embedding.T)
        similarities = similarities.flatten()
        
        # Get top k documents
        top_k = RAG_CONFIG["max_documents"]
        top_indices = np.argsort(similarities)[-top_k:]
        
        return [self.documents[i] for i in top_indices]

    def _sparse_retrieval(self, query: str) -> List[str]:
        """Perform sparse retrieval using BM25."""
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k documents
        top_k = RAG_CONFIG["max_documents"]
        top_indices = np.argsort(doc_scores)[-top_k:]
        
        return [self.documents[i] for i in top_indices]

    def _dense_retrieval(self, query: str) -> List[str]:
        """Perform dense retrieval using embeddings."""
        # Generate query embedding
        query_embedding = self._generate_embeddings([query])[0]
        
        # Retrieve relevant documents
        return self._retrieve_documents(query_embedding)

    def _hybrid_retrieval(self, query: str) -> List[str]:
        """Perform hybrid retrieval combining sparse and dense methods."""
        # Get documents from both methods
        sparse_docs = set(self._sparse_retrieval(query))
        dense_docs = set(self._dense_retrieval(query))
        
        # Combine results
        hybrid_docs = list(sparse_docs.union(dense_docs))
        
        # Limit to max documents
        return hybrid_docs[:RAG_CONFIG["max_documents"]]

    async def _generate_response(self, query: str, contexts: List[str]) -> str:
        """Generate a response using the LLM with the retrieved contexts."""
        combined_context = " ".join(contexts)
        system_prompt = """You are a helpful AI assistant using RAG to answer queries.
        Use the retrieved documents to answer the user's query accurately and concisely."""
        
        prompt = f"Retrieved Documents:\n{combined_context}\nQuery: {query}"
        
        try:
            response = await self.llm.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.llm_config.get('temperature', 0.7),
                max_tokens=self.llm_config.get('max_tokens', None)
            )
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
