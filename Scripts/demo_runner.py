"""
Script to demonstrate the CAG Framework in action.
"""

import asyncio
import logging
import time
from pathlib import Path

from cag_demo.cag_framework import CAGFramework
from cag_demo.rag_framework import RAGFramework
from cag_demo.config import DATA_DIR, LOGS_DIR

# Create necessary directories
for dir_path in [DATA_DIR / "Preloaded_Contexts", 
                DATA_DIR / "Retrieved_Documents",
                LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def run_demo():
    """Run a demonstration comparing CAG and RAG frameworks."""
    
    # Initialize frameworks
    cag = CAGFramework('gpt4', DATA_DIR / "Preloaded_Contexts")
    rag = RAGFramework('gpt4', DATA_DIR / "Retrieved_Documents")
    
    # Test queries
    queries = [
        "What are the main advantages of the CAG framework?",
        "How does CAG compare to traditional RAG in terms of performance?",
        "Explain how CAG eliminates retrieval steps.",
    ]
    
    logger.info("Starting CAG vs RAG demonstration...")
    
    for i, query in enumerate(queries, 1):
        logger.info(f"\nQuery {i}: {query}")
        
        # Test CAG
        logger.info("\nCAG Response:")
        start_time = time.time()
        cag_response, cag_metrics = await cag.query(query)
        cag_time = time.time() - start_time
        
        logger.info(f"Response: {cag_response}")
        logger.info(f"Time taken: {cag_time:.2f} seconds")
        logger.info(f"Metrics: {cag_metrics}")
        
        # Test RAG
        logger.info("\nRAG Response:")
        start_time = time.time()
        rag_response, rag_metrics = await rag.query(query)
        rag_time = time.time() - start_time
        
        logger.info(f"Response: {rag_response}")
        logger.info(f"Time taken: {rag_time:.2f} seconds")
        logger.info(f"Metrics: {rag_metrics}")
        
        # Compare results
        logger.info("\nComparison:")
        logger.info(f"CAG vs RAG time difference: {rag_time - cag_time:.2f} seconds")
        logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_demo())
