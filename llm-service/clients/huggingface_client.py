from huggingface_hub import InferenceClient
from typing import Dict

class HuggingFaceClient:
    def __init__(self, api_key: str):
        self.client = InferenceClient(token=api_key, timeout=60)

    def query(self, prompt: str, model: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> Dict:
        try:
            # Use chat completions API for conversational models
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            # Extract the generated text from response
            generated_text = completion.choices[0].message.content
            
            return {
                "model": model,
                "response": generated_text
            }
        except Exception as e:
            raise Exception(f"HuggingFace API error: {str(e)}")