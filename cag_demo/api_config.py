"""
API configuration and key management for the CAG Demonstrator Agent.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

class APIConfig:
    """Manages API configurations and keys for different LLM providers."""
    
    @staticmethod
    def get_openai_config():
        """Get OpenAI API configuration."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        return {
            'api_key': api_key,
            'base_url': 'https://api.openai.com/v1',
            'model': 'gpt-4'
        }
    
    @staticmethod
    def get_anthropic_config():
        """Get Anthropic API configuration."""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key not found in environment variables")
        return {
            'api_key': api_key,
            'model': 'claude-3-sonnet'
        }
    
    @staticmethod
    def get_mistral_config():
        """Get Mistral API configuration."""
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("Mistral API key not found in environment variables")
        return {
            'api_key': api_key,
            'model': 'mistral-large'
        }
    
    @staticmethod
    def get_groq_config():
        """Get Groq API configuration."""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("Groq API key not found in environment variables")
        return {
            'api_key': api_key,
            'model': 'mixtral-8x7b'
        }
    
    @staticmethod
    def get_google_config():
        """Get Google API configuration."""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key not found in environment variables")
        return {
            'api_key': api_key,
            'model': 'gemini-pro'
        }
    
    @classmethod
    def get_config_for_llm(cls, llm_name: str) -> dict:
        """
        Get API configuration for a specific LLM.
        
        Args:
            llm_name: Name of the LLM (e.g., 'gpt4', 'claude', etc.)
            
        Returns:
            dict: API configuration for the specified LLM
        """
        config_map = {
            'gpt4': cls.get_openai_config,
            'claude': cls.get_anthropic_config,
            'mistral': cls.get_mistral_config,
            'groq': cls.get_groq_config,
            'gemini': cls.get_google_config
        }
        
        if llm_name not in config_map:
            raise ValueError(f"Unknown LLM: {llm_name}")
        
        return config_map[llm_name]()
