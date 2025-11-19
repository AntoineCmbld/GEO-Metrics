from groq import Groq
from typing import Dict

class GroqClient:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def query(self, prompt: str) -> Dict:
        response = self.client.chat.completions.create(
            model="llama2-70b-4096",
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "provider": "groq",
            "response": response.choices[0].message.content,
            "score": 0.9  # Placeholder score
        }