import requests


class LLMService:
    """Service for working with LLM (Google Gemini)"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")
            
        self.api_key = api_key
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer based on context"""
        
        if not question.strip():
            raise ValueError("question cannot be empty")
            
        if not context.strip():
            raise ValueError("context cannot be empty")
        
        prompt = f"""You are an expert consultant. Answer the question using ONLY information from the provided context.

Rules:
Use only facts from the context
If the context doesn't contain information to answer, say so honestly
Be precise and specific

Contex:
{context}

Question: {question}

Answer:"""

        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if ('candidates' not in result or 
                not result['candidates'] or 
                'content' not in result['candidates'][0] or
                'parts' not in result['candidates'][0]['content'] or
                not result['candidates'][0]['content']['parts']):
                raise Exception("invalid response format")
            
            answer = result['candidates'][0]['content']['parts'][0]['text']
            
            if not answer.strip():
                raise Exception("empty response")
                
            return answer
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    raise Exception(f"gemini API error: {error_detail}")
                except:
                    raise Exception(f"gemini API error: {e.response.status_code} - {e.response.text}")
            else:
                raise Exception(f"gemini API error: {str(e)}")
        except Exception as e:
            if "API error" in str(e) or "connection error" in str(e):
                raise e
            else:
                raise Exception(f"gen error: {str(e)}")