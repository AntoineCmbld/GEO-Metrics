import os
from openai import OpenAI
import google.genai as genai
from groq import Groq

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
google_client = genai.Client(api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

async def translate_prompt(prompt: str, country: str) -> str:
    translation_prompt = f"Translate the following text into the main language spoken in {country}:\n{prompt}"
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": translation_prompt}],
        max_tokens=500
    )
    return resp.choices[0].message.content

async def generate_ai_response(prompt: str, provider: str, country: str = None) -> str:
    if country:
        prompt = await translate_prompt(prompt, country)
        prompt += f" in {country}"

    provider = provider.lower()

    if provider == "openai":
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return resp.choices[0].message.content

    elif provider == "google_gemini":
        resp = google_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        return resp.text

    elif provider == "groq":
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unsupported provider: {provider}")
