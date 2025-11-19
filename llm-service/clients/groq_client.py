from groq import Groq
from typing import Dict

class GroqClient:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def query(self, prompt: str, model: str = "llama-3.1-8b-instant") -> Dict:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "provider": "groq",
            "model": model,
            "response": response.choices[0].message.content
        }