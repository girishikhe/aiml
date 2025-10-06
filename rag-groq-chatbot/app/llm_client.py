import os
import requests
from utils.logger import get_logger
from utils.custom_exceptions import LLMResponseError, WebSearchError

logger = get_logger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/v1/chat/completions")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL = os.getenv("TAVILY_SEARCH_URL", "https://api.tavily.com/search")


def groq_call(prompt: str, max_tokens: int = 400, temperature: float = 0.2) -> str:
    """Calls Groq API with a prompt and returns completion text."""
    if not GROQ_API_KEY:
        raise LLMResponseError("Missing GROQ_API_KEY in .env")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "mixtral-8x7b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info("Groq call successful.")
        return text.strip()

    except Exception as e:
        logger.exception("Groq API call failed.")
        raise LLMResponseError(f"Groq API call failed: {e}") from e


def web_search_tavily(query: str, max_results: int = 3):
    """Performs a web search using the Tavily API."""
    if not TAVILY_API_KEY:
        logger.warning("Missing Tavily API key. Skipping web search.")
        return []

    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {"query": query, "max_results": max_results}

    try:
        response = requests.post(TAVILY_SEARCH_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        if "results" not in data:
            raise WebSearchError("Invalid Tavily API response format")

        logger.info(f"Tavily search successful for query: '{query}'")
        return data["results"]

    except Exception as e:
        logger.exception("Tavily web search failed.")
        raise WebSearchError(f"Tavily web search failed: {e}") 
