import openai
from typing import Dict

class ChatGPTClient:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def query(self, prompt: str) -> Dict:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "provider": "openai",
            "response": response.choices[0].message.content,
            "score": 0.8  # Placeholder score
        }