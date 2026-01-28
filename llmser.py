import requests


class LLMService:
    """Minimal Gemini API client with clean architecture."""

    API_URL = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
    )

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")

        self.api_key = api_key


    def generate_answer(self, question: str, context: str) -> str:
        self._validate(question, context)

        response = self._request(
            prompt=self._build_prompt(question, context)
        )

        return self._parse_response(response)


    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        return f"""
You are an expert assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say: "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
""".strip()


    @staticmethod
    def _validate(question: str, context: str) -> None:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not context.strip():
            raise ValueError("Context cannot be empty")


    def _request(self, prompt: str) -> dict:
        response = requests.post(
            url=f"{self.API_URL}?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json=self._payload(prompt),
            timeout=20,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Gemini API error: {response.text}")

        return response.json()

    @staticmethod
    def _payload(prompt: str) -> dict:
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1024,
            },
        }


    @staticmethod
    def _parse_response(result: dict) -> str:
        try:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise RuntimeError("Invalid Gemini response format")

        if not text.strip():
            raise RuntimeError("Empty Gemini response")

        return text.strip()
