# """
# llm_client.py — LLM client using OpenAI library → Gemini free API.

# Uses the openai Python package (>=1.0) with Gemini's OpenAI-compatible
# base URL.  This way the rest of the code uses standard OpenAI patterns
# while calling Gemini models for free.

# Free models (https://aistudio.google.com/apikey):
#   gemini-2.0-flash-lite  ← fastest, most generous free quota
#   gemini-2.0-flash       ← smarter, moderately generous
#   gemini-1.5-flash       ← fallback alternative

# Usage:
#   from llm_client import chat, FAST_MODEL, SMART_MODEL
#   response = chat(messages, model=SMART_MODEL)
# """
# import os
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# # Two-tier model strategy:
# #   FAST_MODEL  — simple tasks (classification, routing) → saves quota
# #   SMART_MODEL — complex reasoning (triage, act, reply) → better quality
# FAST_MODEL  = os.getenv("GEMINI_FAST_MODEL",  "gemini-2.0-flash-lite")
# SMART_MODEL = os.getenv("GEMINI_SMART_MODEL", "gemini-2.0-flash")


# def get_client() -> OpenAI:
#     api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         raise EnvironmentError(
#             "GEMINI_API_KEY not set. "
#             "Get a free key at https://aistudio.google.com/apikey"
#         )
#     return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)


# def chat(
#     messages: list[dict],
#     model: str = FAST_MODEL,
#     temperature: float = 0.0,
#     max_tokens: int = 1024,
# ) -> str:
#     """Call Gemini via OpenAI-compatible endpoint. Returns assistant text."""
#     client = get_client()
#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=temperature,
#         max_tokens=max_tokens,
#     )
#     return response.choices[0].message.content.strip()



# llm_client.py (for Groq)
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Update .env: GROQ_API_KEY=your_key
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1", # Groq's base URL
)

FAST_MODEL = "llama-3.1-8b-instant"
SMART_MODEL = "llama-3.3-70b-versatile"

def chat(messages: list[dict], model: str = FAST_MODEL, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()