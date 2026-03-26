import os
import hashlib
import json
from dotenv import load_dotenv

load_dotenv()

# ── Groq setup ────────────────────────────────────────────────
from groq import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")

# ── Gemini setup (new SDK) ────────────────────────────────────
from google import genai
from google.genai import types
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL  = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── Answer cache (persists to disk) ──────────────────────────
CACHE_FILE = "answer_cache.json"

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def cache_key(content: str, query: str) -> str:
    """Short hash of query + content as cache key."""
    raw = f"{query.strip().lower()}::{content[:500]}"
    return hashlib.md5(raw.encode()).hexdigest()

# ─────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────
def build_prompt(content: str, query: str) -> str:
    return f"""
You are a research assistant.

Use ONLY the provided research paper content to answer the question.

Rules:
1. If the answer exists in the provided papers, generate the answer.
2. Mention ONLY the title of the research paper used for the answer.
3. If the answer is not present in the papers, respond exactly:

Answer: Not found in the retrieved papers.
Research Paper: None

Response format:

Answer:
<answer>

Research Paper:
<paper title>, <paper title> if multiple papers are used

Content:
{content}

Question:
{query}
"""

# ─────────────────────────────────────────────────────────────
# GROQ CALL
# ─────────────────────────────────────────────────────────────
def ask_groq_llm(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful and precise research assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1024
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────────────────────────
# GEMINI CALL
# ─────────────────────────────────────────────────────────────
def ask_gemini_llm(prompt: str) -> str:
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=1024,
            system_instruction="You are a helpful and precise research assistant."
        )
    )
    return response.text

# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────
def ask_groq(content: str, query: str) -> str:
    # ── Check cache first — no API call needed ───────────────
    cache = load_cache()
    key   = cache_key(content, query)

    if key in cache:
        print("⚡ Returning cached answer — no API call made")
        return cache[key]

    prompt = build_prompt(content, query)
    answer = None

    # ── Try Groq first ───────────────────────────────────────
    try:
        print("🤖 Using Groq...")
        answer = ask_groq_llm(prompt)
        print("✅ Groq responded")

    except Exception as groq_error:
        groq_msg = str(groq_error)
        is_rate_limit = "429" in groq_msg or "rate_limit" in groq_msg.lower()

        if is_rate_limit:
            wait_hint = ""
            if "Please try again in" in groq_msg:
                try:
                    wait_hint = groq_msg.split("Please try again in")[1].split(".")[0].strip()
                    wait_hint = f" (resets in {wait_hint})"
                except Exception:
                    pass
            print(f"⚠️ Groq rate limit{wait_hint} — switching to Gemini...")
        else:
            print(f"⚠️ Groq error: {groq_msg[:120]} — switching to Gemini...")

        # ── Fallback to Gemini ───────────────────────────────
        try:
            print("🤖 Using Gemini...")
            answer = f"[Answered by Gemini]\n{ask_gemini_llm(prompt)}"
            print("✅ Gemini responded")

        except Exception as gemini_error:
            gemini_msg = str(gemini_error)
            print(f"❌ Gemini also failed: {gemini_msg[:120]}")

            if "429" in gemini_msg or "quota" in gemini_msg.lower():
                return (
                    "Answer: Both Groq and Gemini are currently rate-limited. "
                    "Please wait a few minutes and try again.\n"
                    "Research Paper: None"
                )
            return (
                f"Answer: Both AI providers failed.\n"
                f"Groq: {str(groq_error)[:100]}\n"
                f"Gemini: {gemini_msg[:100]}\n"
                "Research Paper: None"
            )

    # ── Cache the successful answer ──────────────────────────
    if answer:
        cache[key] = answer
        save_cache(cache)

    return answer