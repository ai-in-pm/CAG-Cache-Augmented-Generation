"""
Configuration settings for the CAG Demonstrator Agent.
"""

from pathlib import Path
from typing import Dict, List

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
LOGS_DIR = BASE_DIR / "Logs"

# LLM configurations
LLM_CONFIGS = {
    "gpt4": {
        "name": "ChatGPT-4o",
        "model": "gpt-4",
        "max_tokens": 8192,
        "temperature": 0.7,
    },
    "claude": {
        "name": "Claude Sonnet 3.5",
        "model": "claude-3-sonnet",
        "max_tokens": 4096,
        "temperature": 0.7,
    },
    "gemini": {
        "name": "Gemini",
        "model": "gemini-pro",
        "max_tokens": 8192,
        "temperature": 0.7,
    },
    "mistral": {
        "name": "Mistral",
        "model": "mistral-large",
        "max_tokens": 4096,
        "temperature": 0.7,
    },
    "groq": {
        "name": "Groq",
        "model": "mixtral-8x7b",
        "max_tokens": 4096,
        "temperature": 0.7,
    }
}

# CAG Framework settings
CAG_CONFIG = {
    "cache_size": 1000000,  # Number of key-value pairs to cache
    "context_window": 8192,  # Maximum context window size
    "preload_chunk_size": 512,  # Size of chunks for preloading
}

# RAG Framework settings
RAG_CONFIG = {
    "retriever_type": "hybrid",  # Options: sparse, dense, hybrid
    "max_documents": 5,  # Maximum number of documents to retrieve
    "sparse_config": {
        "type": "bm25",
        "b": 0.75,
        "k1": 1.5,
    },
    "dense_config": {
        "model": "text-embedding-3-small",
        "dimensions": 1536,
    }
}

# Evaluation metrics
METRICS = [
    "response_time",
    "accuracy",
    "bert_score",
    "memory_usage",
    "system_complexity"
]

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "cag_demonstrator.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["file"],
            "level": "INFO",
            "propagate": True
        },
    }
}
