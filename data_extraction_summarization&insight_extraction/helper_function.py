import torch 
from groq import Groq
from dotenv import load_dotenv
import os, json

load_dotenv()

def summeriser(text, tokenizer, model):
    words = text.split()
    chunk_size = 600
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    full_summary = []
    
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=250, 
                min_length=50,     
                length_penalty=0.8,
                num_beams=4,
                early_stopping=True
            )
            
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_summary.append(summary)
        
    final_summary = " ".join(full_summary)
    print("summary length: ", len(final_summary.split()), "words")
    return final_summary


def insigth_extraction(summary):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
    Extract structured insights from the text below.

    Return ONLY valid JSON.
    Do not add explanations.
    Do not add markdown.
    Do not add text before or after JSON.

    Use this exact format:

    {{
    "domain": [],
    "research_problem": "",
    "methods": [],
    "datasets": [],
    "metrics": [],
    "key_findings": "",
    "limitations": "",
    "future_directions": ""
    }}

    Abstract/Text:
    {summary}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print("Failed to decode JSON from Groq.")
        return None