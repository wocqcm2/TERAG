"""
LLM client for TERAG framework.
Provides unified interface for OpenAI-compatible APIs.
"""

import time
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMClient:
    """
    LLM client with unified interface for OpenAI-compatible APIs.
    Supports both DeepInfra and OpenAI providers.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout: int = 260,
        max_retries: int = 5
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for API endpoint
            model_name: Model name to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
        """
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Load prompts
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompt templates from prompts.yaml"""
        possible_paths = [
            Path("config/prompts.yaml"),
            Path("../config/prompts.yaml"),
            Path(__file__).parent.parent.parent / "config" / "prompts.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
        
        # Return default prompts if file not found
        return {
            "concept_extraction": {
                "system_message": "Extract named entities (NER) and document-level concepts. Output only the core content in structured format.",
                "passage_prompt": ""
            },
            "ner": {
                "system_message": "You are a named entity recognition system. Extract only the key entities from the given text.",
                "user_prompt": "Extract the key named entities from the following question. Return only the entity names, separated by commas.\n\nQuestion: {query}\n\nEntities:"
            },
            "answer_generation": {
                "system_message": "You are a helpful assistant that answers questions based on the provided context.",
                "user_prompt": "Answer the following question based on the provided context. Be concise and accurate.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            }
        }
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30)
    )
    def _single_request(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
        return_usage: bool = False
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Make a single API request with retry logic.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_usage: Whether to return usage statistics
            
        Returns:
            Generated text, or tuple of (text, usage_dict) if return_usage=True
        """
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            timeout=self.timeout
        )
        
        time_cost = time.time() - start_time
        content = response.choices[0].message.content
        
        if return_usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "time": time_cost
            }
            return content, usage
        else:
            return content
    
    def generate_response(
        self,
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
        return_text_only: bool = True,
        max_workers: int = 3,
        **kwargs
    ) -> Union[List[str], List[Tuple[str, Dict[str, Any]]]]:
        """
        Generate responses for single or batch messages.
        
        Args:
            messages: Single message list or batch of message lists
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_text_only: If True, return only text; if False, return (text, usage)
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of generated texts or list of (text, usage) tuples
        """
        # Determine if batch or single
        is_batch = isinstance(messages[0], list)
        if not is_batch:
            messages = [messages]
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            def process_single(msg_list):
                try:
                    return self._single_request(
                        msg_list,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        return_usage=not return_text_only
                    )
                except Exception as e:
                    print(f"Request failed: {e}")
                    if return_text_only:
                        return ""
                    else:
                        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "time": 0.0}
            
            futures = [executor.submit(process_single, msg) for msg in messages]
            results = [future.result() for future in futures]
        
        return results
    
    def ner(self, query: str) -> str:
        """
        Extract named entities from a query using NER prompt.
        
        Args:
            query: Input query text
            
        Returns:
            Comma-separated entity names
        """
        system_msg = self.prompts["ner"]["system_message"]
        user_msg = self.prompts["ner"]["user_prompt"].format(query=query)
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        try:
            response = self._single_request(
                messages,
                max_new_tokens=512,
                temperature=0.1,
                return_usage=False
            )
            return response.strip()
        except Exception as e:
            print(f"NER extraction failed: {e}")
            return ""
    
    def generate_with_context(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.5
    ) -> str:
        """
        Generate answer based on context and question.
        
        Args:
            question: Question to answer
            context: Context passages
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated answer
        """
        system_msg = self.prompts["answer_generation"]["system_message"]
        user_msg = self.prompts["answer_generation"]["user_prompt"].format(
            context=context,
            question=question
        )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        try:
            response = self._single_request(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                return_usage=False
            )
            return response.strip()
        except Exception as e:
            print(f"Answer generation failed: {e}")
            return ""

