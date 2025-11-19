import openai
from typing import Dict

class ChatGPTClient:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def query(self, prompt: str, model: str = "gpt-3.5-turbo") -> Dict:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "provider": "openai",
            "model": model,
            "response": response.choices[0].message.content
        }