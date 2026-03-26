from google import genai
import os
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def ask_gemini(content, query):
    
    # agumented 
    prompt = f"""
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
        <paper title>, <paper title> if multiple paper are there

        Content:
        {content}

        Question:
        {query}
        """
    
    # retrival 
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.9
        ),
    )
    print(response.text)
    
    return response.text