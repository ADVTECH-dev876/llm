

import os
import logging
from typing import List, Dict, Optional, Generator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai
from openai import OpenAI, APIError, RateLimitError, APIConnectionError, AuthenticationError
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIService:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY env var)")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError))
    )
    def _make_api_call(self, messages: List[Dict[str, str]], **kwargs):
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
                "top_p": kwargs.get("top_p", 1.0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0),
            }
            params = {k: v for k, v in params.items() if v is not None}
            return self.client.chat.completions.create(**params)
        except (AuthenticationError, APIError) as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError))
    )
    def _make_streaming_api_call(self, messages: List[Dict[str, str]], **kwargs):
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
                "stream": True,
            }
            params = {k: v for k, v in params.items() if v is not None}
            stream = self.client.chat.completions.create(**params)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except (AuthenticationError, APIError) as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def generate_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        messages = [{"role": "system", "content": system_prompt or "You are a helpful assistant."}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        response = self._make_api_call(messages, **kwargs)
        return response.choices[0].message.content.strip()

    def stream_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        messages = [{"role": "system", "content": system_prompt or "You are a helpful assistant."}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        yield from self._make_streaming_api_call(messages, **kwargs)


# Example usage (remove before production)
if __name__ == "__main__":
    service = OpenAIService()
    print(service.generate_response("Explain quantum computing in simple terms.")) 
//OPENAI_API_KEY=your_api_key_here
//pip install openai python-dotenv tenacity
