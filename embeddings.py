import hashlib
from pathlib import Path

import numpy as np
import requests


class EmbeddingService:
    """Clean embedding client with caching support."""

    API_URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, cache_dir: str = "embeddings_cache"):
        if not api_key:
            raise ValueError("API key is required")

        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_embedding(self, text: str, model: str = "voyage-2") -> np.ndarray:
        if not text.strip():
            raise ValueError("Text cannot be empty")

        cache_path = self._cache_path(text, model)

        embedding = self._load_cache(cache_path)
        if embedding is not None:
            return embedding

        embedding = self._request_embedding(text, model)
        self._save_cache(cache_path, embedding)

        return embedding


    def _cache_path(self, text: str, model: str) -> Path:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}_{model}.npy"

    @staticmethod
    def _load_cache(path: Path) -> np.ndarray | None:
        if not path.exists():
            return None

        try:
            return np.load(path)
        except Exception:
            path.unlink(missing_ok=True)
            return None

    @staticmethod
    def _save_cache(path: Path, embedding: np.ndarray) -> None:
        try:
            np.save(path, embedding)
        except Exception:
            pass

    def _request_embedding(self, text: str, model: str) -> np.ndarray:
        response = requests.post(
            self.API_URL,
            headers=self._headers(),
            json={"input": [text], "model": model},
            timeout=20,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Voyage API error: {response.text}")

        data = response.json().get("data")
        if not data:
            raise RuntimeError("Invalid API response")

        return np.array(data[0]["embedding"], dtype=np.float32)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }