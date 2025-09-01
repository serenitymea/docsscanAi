import hashlib
import numpy as np
import requests
from pathlib import Path


class EmbeddingService:
    """Service for working with embeddings"""
    
    def __init__(self, api_key: str, cache_dir: str = "embeddings_cache"):
        if not api_key:
            raise ValueError("API key is required")
            
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.api_url = "https://api.voyageai.com/v1/embeddings"
        
    def get_embedding(self, text: str, model: str = "voyage-2") -> np.ndarray:
        """Get embedding with caching"""
        if not text.strip():
            raise ValueError("Text cannot be empty")

        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_file = self.cache_dir / f"{text_hash}_{model}.npy"

        if cache_file.exists():
            try:
                print(f"Loading from cache: {text[:50]}...")
                return np.load(cache_file)
            except Exception as e:
                print(f"Cache file corrupted, regenerating: {str(e)}")
                cache_file.unlink(missing_ok=True)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "input": [text],
            "model": model
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'data' not in result or not result['data']:
                raise Exception("Invalid API response format")
                
            embedding = np.array(result['data'][0]['embedding'])

            try:
                np.save(cache_file, embedding)
                print(f"Created embedding: {text[:50]}...")
            except Exception as e:
                print(f"Warning: Could not save to cache: {str(e)}")
            
            return embedding
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    raise Exception(f"Voyage AI API error: {error_detail}")
                except:
                    raise Exception(f"Voyage AI API error: {e.response.status_code} - {e.response.text}")
            else:
                raise Exception(f"Voyage AI API connection error: {str(e)}")
        except Exception as e:
            raise Exception(f"Embedding generation error: {str(e)}")