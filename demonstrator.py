"""
CAG Demonstrator Agent for comparing Cache-Augmented Generation (CAG) and RAG frameworks.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from cag_demo.config import (
    DATA_DIR,
    LOGS_DIR,
    CAG_CONFIG,
    RAG_CONFIG,
    LLM_CONFIGS
)
from cag_demo.cag_framework import CAGFramework
from cag_demo.rag_framework import RAGFramework

# Create necessary directories
RESULTS_DIR = Path(__file__).parent / "Results"
for dir_path in [DATA_DIR / "Preloaded_Contexts", 
                DATA_DIR / "Retrieved_Documents",
                LOGS_DIR,
                RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'demonstrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CAGDemonstrator:
    """Agent for demonstrating CAG Framework capabilities."""
    
    def __init__(self, llm_name: str = 'gpt4'):
        """
        Initialize the demonstrator.
        
        Args:
            llm_name: Name of the LLM to use
        """
        self.llm_name = llm_name
        self.cag = CAGFramework(llm_name, DATA_DIR / "Preloaded_Contexts")
        self.rag = RAGFramework(llm_name, DATA_DIR / "Retrieved_Documents")
        self.results = []
    
    async def compare_frameworks(self, query: str) -> dict:
        """
        Compare CAG and RAG frameworks on a given query.
        
        Args:
            query: Question to ask both frameworks
            
        Returns:
            dict: Comparison metrics
        """
        logger.info(f"\nProcessing query: {query}")
        
        # Test CAG
        logger.info("\nCAG Response:")
        start_time = time.time()
        cag_response, cag_metrics = await self.cag.query(query)
        cag_time = time.time() - start_time
        
        logger.info(f"Response: {cag_response}")
        logger.info(f"Time taken: {cag_time:.2f} seconds")
        logger.info(f"Metrics: {cag_metrics}")
        
        # Test RAG
        logger.info("\nRAG Response:")
        start_time = time.time()
        rag_response, rag_metrics = await self.rag.query(query)
        rag_time = time.time() - start_time
        
        logger.info(f"Response: {rag_response}")
        logger.info(f"Time taken: {rag_time:.2f} seconds")
        logger.info(f"Metrics: {rag_metrics}")
        
        # Compare results
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'cag_response': cag_response,
            'rag_response': rag_response,
            'cag_time': cag_time,
            'rag_time': rag_time,
            'time_difference': rag_time - cag_time,
            'cag_metrics': cag_metrics,
            'rag_metrics': rag_metrics
        }
        
        self.results.append(comparison)
        
        logger.info("\nComparison:")
        logger.info(f"CAG vs RAG time difference: {comparison['time_difference']:.2f} seconds")
        logger.info("=" * 80)
        
        return comparison
    
    def save_results(self):
        """Save results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = RESULTS_DIR / f"comparison_results_{timestamp}.json"
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'llm_name': self.llm_name,
            'cag_config': CAG_CONFIG,
            'rag_config': RAG_CONFIG,
            'results': self.results
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_file}")

async def main():
    """Run the demonstration."""
    demonstrator = CAGDemonstrator()
    
    # Test queries
    queries = [
        "What are the main advantages of the CAG framework?",
        "How does CAG compare to traditional RAG in terms of performance?",
        "Explain how CAG eliminates retrieval steps.",
        "What are the key differences between CAG and RAG architectures?",
        "How does CAG handle context management differently from RAG?"
    ]
    
    logger.info("Starting CAG Framework Demonstration...")
    
    for query in queries:
        await demonstrator.compare_frameworks(query)
    
    # Save results
    demonstrator.save_results()

if __name__ == "__main__":
    asyncio.run(main())
