import re
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import httpx
from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import DocumentChunk
from app.services.guardrails import EXACT_REFUSAL_PHRASE

class BaseLLMProvider(ABC):
    """Abstract base class for LLM generation engines."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, retrieved_chunks: Optional[List[Tuple[DocumentChunk, float]]] = None) -> str:
        """Generates an answer given system prompt and user context."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Returns the identifier of the active model and provider."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the provider endpoint is healthy and accessible."""
        pass


class OllamaProvider(BaseLLMProvider):
    """Local inference provider via Ollama (e.g. Llama 3 / Mistral) for GDPR compliance."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL

    def is_available(self) -> bool:
        """Checks if local Ollama daemon is running."""
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def generate(self, system_prompt: str, user_prompt: str, retrieved_chunks: Optional[List[Tuple[DocumentChunk, float]]] = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,  # Zero temperature for deterministic grounding
                "top_p": 0.9,
            },
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                return data.get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation request failed ({e}).")
            raise RuntimeError(f"Ollama generation failed: {e}")

    def get_model_name(self) -> str:
        return f"Ollama ({self.model})"


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def generate(self, system_prompt: str, user_prompt: str, retrieved_chunks: Optional[List[Tuple[DocumentChunk, float]]] = None) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not configured.")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        with httpx.Client(timeout=45.0) as client:
            res = client.post(url, headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
            return data["choices"][0]["message"]["content"].strip()

    def get_model_name(self) -> str:
        return f"OpenAI ({self.model})"


class DeterministicGroundedProvider(BaseLLMProvider):
    """
    High-fidelity deterministic extractive QA provider.
    Ensures that evaluations, demos, and CI/CD pipelines run flawlessly
    without external API keys or GPU daemons while maintaining strict grounding.
    """

    def is_available(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return "Deterministic Grounded Synthesizer (Local Fallback)"

    def generate(self, system_prompt: str, user_prompt: str, retrieved_chunks: Optional[List[Tuple[DocumentChunk, float]]] = None) -> str:
        if not retrieved_chunks:
            return EXACT_REFUSAL_PHRASE

        # Extract user question from prompt
        match = re.search(r"User Question:\s*(.+?)(?:\n\n|$)", user_prompt)
        question = match.group(1).strip() if match else ""
        query_words = set(re.findall(r"\w+", question.lower()))

        # Remove stop words
        stopwords = {"what", "when", "how", "why", "where", "is", "are", "the", "a", "an", "for", "to", "in", "of", "and", "do", "does", "wie", "was", "wann", "wo", "ist", "sind", "der", "die", "das", "ein", "eine", "fuer", "für"}
        keywords = query_words - stopwords

        if not keywords:
            keywords = query_words

        # Find best matching sentences in retrieved chunks
        best_sentences: List[Tuple[str, str, int, int]] = []
        for chunk, score in retrieved_chunks:
            # Split into clean sentences
            sentences = re.split(r"(?<=[.!?])\s+", chunk.content)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 15:
                    continue
                s_words = set(re.findall(r"\w+", s_clean.lower()))
                overlap = len(keywords & s_words)
                if overlap > 0:
                    best_sentences.append((s_clean, chunk.document_name, chunk.page_number, overlap))

        if not best_sentences:
            return EXACT_REFUSAL_PHRASE

        # Sort by keyword overlap descending
        best_sentences.sort(key=lambda x: x[3], reverse=True)
        top_sentences = best_sentences[:3]

        answer_lines = []
        for text, doc, page, _ in top_sentences:
            answer_lines.append(f"{text} [{doc}, Page {page}]")

        return "\n\n".join(answer_lines)


class LLMFactory:
    """Factory to instantiate the appropriate LLM provider."""

    @classmethod
    def get_provider(cls) -> BaseLLMProvider:
        mode = settings.LLM_PROVIDER.lower()

        if mode == "ollama":
            ollama = OllamaProvider()
            if ollama.is_available():
                logger.info(f"Using local Ollama provider with model '{settings.OLLAMA_MODEL}'.")
                return ollama
            logger.warning("Configured for Ollama, but Ollama service is not responding. Falling back.")

        elif mode == "openai":
            openai_p = OpenAIProvider()
            if openai_p.is_available():
                logger.info(f"Using OpenAI provider with model '{settings.OPENAI_MODEL}'.")
                return openai_p
            logger.warning("Configured for OpenAI, but OPENAI_API_KEY is not set. Falling back.")

        elif mode == "mock":
            logger.info("Using Deterministic Grounded Provider (Mock mode).")
            return DeterministicGroundedProvider()

        # "auto" detection mode:
        # Check Ollama first (local privacy preference for European enterprise)
        ollama = OllamaProvider()
        if ollama.is_available():
            logger.info(f"Auto-detected active local Ollama service. Using {ollama.get_model_name()}.")
            return ollama

        # Check OpenAI if key is present
        openai_p = OpenAIProvider()
        if openai_p.is_available():
            logger.info(f"Auto-detected OpenAI API key. Using {openai_p.get_model_name()}.")
            return openai_p

        logger.info("No active Ollama or OpenAI key detected. Using high-fidelity Deterministic Grounded Synthesizer.")
        return DeterministicGroundedProvider()
