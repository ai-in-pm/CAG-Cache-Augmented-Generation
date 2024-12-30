"""
Interface for interacting with various LLM providers.
"""

from typing import Dict, Optional
import openai
from anthropic import Anthropic
from mistralai.client import MistralClient
import google.generativeai as genai
import httpx

from cag_demo.api_config import APIConfig

class LLMInterface:
    """Interface for interacting with different LLM providers."""
    
    def __init__(self, llm_name: str):
        """
        Initialize LLM interface.
        
        Args:
            llm_name: Name of the LLM to use
        """
        self.llm_name = llm_name
        self.config = APIConfig.get_config_for_llm(llm_name)
        self._setup_client()
    
    def _setup_client(self):
        """Set up the appropriate client based on LLM type."""
        if self.llm_name == 'gpt4':
            openai.api_key = self.config['api_key']
            self.client = openai.OpenAI()
        elif self.llm_name == 'claude':
            self.client = Anthropic(api_key=self.config['api_key'])
        elif self.llm_name == 'mistral':
            self.client = MistralClient(api_key=self.config['api_key'])
        elif self.llm_name == 'groq':
            self.client = httpx.Client(
                base_url="https://api.groq.com/v1",
                headers={"Authorization": f"Bearer {self.config['api_key']}"}
            )
        elif self.llm_name == 'gemini':
            genai.configure(api_key=self.config['api_key'])
            self.client = genai.GenerativeModel('gemini-pro')
    
    async def generate_response(self, 
                              prompt: str, 
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            str: Generated response
        """
        try:
            if self.llm_name == 'gpt4':
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.config['model'],
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            elif self.llm_name == 'claude':
                messages = []
                if system_prompt:
                    messages.append({
                        "role": "assistant",
                        "content": system_prompt
                    })
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.messages.create(
                    model=self.config['model'],
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.content[0].text
            
            elif self.llm_name == 'mistral':
                messages = []
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
                messages.append({
                    "role": "user",
                    "content": prompt
                })
                
                response = self.client.chat(
                    model=self.config['model'],
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            elif self.llm_name == 'groq':
                messages = []
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.post(
                    "/chat/completions",
                    json={
                        "model": self.config['model'],
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                )
                return response.json()["choices"][0]["message"]["content"]
            
            elif self.llm_name == 'gemini':
                if system_prompt:
                    prompt = f"{system_prompt}\n\n{prompt}"
                
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens
                    }
                )
                return response.text
            
        except Exception as e:
            raise Exception(f"Error generating response from {self.llm_name}: {str(e)}")
        
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'client') and isinstance(self.client, httpx.Client):
            self.client.close()
