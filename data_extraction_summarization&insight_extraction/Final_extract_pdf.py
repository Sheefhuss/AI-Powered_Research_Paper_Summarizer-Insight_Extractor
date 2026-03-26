import fitz
import re
import os
import json
import uuid
import spacy
from datetime import datetime
from helper_function import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nlp = spacy.load("en_core_web_sm")

# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = text.replace("\r", "\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =========================
# TITLE & AUTHOR EXTRACTION (From data_extraction_PDF.py)
# =========================
def extract_title_and_authors(text):
    abstract_match = re.search(r'\bAbstract\b', text, re.IGNORECASE)
    header_text = text[:abstract_match.start()] if abstract_match else text[:2000]
    lines = header_text.split('\n')
    
    noise_keywords = [
        'issn', 'volume', 'vol', 'issue', 'journal', 'ijisrt', 'arxiv', 
        'copyright', 'department', 'university', 'institute', 'college', 
        'school', 'engineering', 'science', 'technology', 'systems', 'email', '@',
        'scholar', 'professor', 'student', 'received', 'accepted', 'cs.ai',
        'doi', 'https', 'http', 'faculty', 'jl.', 'indonesia', 'java', 'semarang'
    ]

    cleaned_lines = []
    for line in lines:
        cl = line.strip()
        if not cl or len(cl) < 4: continue
        if re.fullmatch(r'[A-Z0-9\-\.\s]+', cl) and len(cl.split()) == 1: continue
        if any(noise in cl.lower() for noise in noise_keywords): continue
        if re.search(r'\d{4}', cl) and len(cl.split()) <= 2: continue
        cleaned_lines.append(cl)

    title_lines = []
    authors = []
    title_done = False
    title_words = [
        'study', 'artificial', 'intelligence', 'analysis', 'model', 'models', 
        'approach', 'system', 'based', 'using', 'effect', 'data', 'consistency', 
        'attack', 'large', 'reasoning', 'multi-turn', 'classification', 
        'recommendation', 'algorithm', 'network', 'learning', 'machine', 
        'cryptography', 'evaluation', 'tree', 'forest', 'movie', 'netflix'
    ]

    for line in cleaned_lines:
        if re.search(r'\b\d{5}\b', line): 
            continue

        doc = nlp(line)
        has_person = any(ent.label_ == "PERSON" for ent in doc.ents)
        has_title_word = any(tw in line.lower() for tw in title_words)
        
        alpha_words = [w for w in line.replace(',', '').split() if w.isalpha()]
        is_capitalized_name = bool(alpha_words) and all(w[0].isupper() for w in alpha_words)
        
        is_author = False
        if has_person and not has_title_word:
            is_author = True
        elif is_capitalized_name and 2 <= len(alpha_words) <= 10 and not has_title_word:
            is_author = True
            
        if is_author:
            title_done = True
            parts = re.split(r',|\band\b', line)
            for p in parts:
                clean_p = re.sub(r'[^a-zA-Z\s\.\-]', '', p).strip()
                if len(clean_p.split()) >= 2:
                    authors.append(clean_p)
        else:
            if not title_done:
                title_lines.append(line)

    title = " ".join(title_lines).strip()
    authors = list(dict.fromkeys(authors))
    return title if title else "Unknown Title", authors

# =========================
# ABSTRACT EXTRACTION (From data_extraction_PDF.py)
# =========================
def extract_abstract(text):
    pattern = re.search(
        r'\bAbstract\b[\s:.]+(.*?)(?=(?:^|\n|(?<=[.!?])\s+)(?:1|I|[IVX]+)?\.?\s*Introduction\b|\n\s*Keywords\b)', 
        text, 
        re.IGNORECASE | re.DOTALL
    )
    if pattern:
        abstract = pattern.group(1).strip()
        return re.sub(r'\s+', ' ', abstract)
        
    fallback = re.search(r'\bAbstract\b[\s:.]+(.{200,1500}?)(?=(?:^|\n|(?<=[.!?])\s+)(?:1|I|[IVX]+)\.)', text, re.IGNORECASE | re.DOTALL)
    if fallback:
        abstract = fallback.group(1).strip()
        return re.sub(r'\s+', ' ', abstract) + "..."
        
    return "Abstract Not Found"

# =========================
# SECTIONS / CONTENT EXTRACTION (From data_extraction_PDF.py)
# =========================
def extract_sections(text):
    ref_match = re.search(r'(?:^|\n|(?<=[.!?])\s+)(?:[IVX]+|\d+)?\.?\s*References\b', text, re.IGNORECASE)
    if ref_match:
        text = text[:ref_match.start()]

    sections = {}
    pattern = r'(?:^|\n|(?<=[.!?])\s+)(?P<id>[IVX]+|\d+(?:\.\d+)?)\.?\s+(?P<title>[A-Z][A-Za-z0-9\s\-\?\:]{2,80})\n'
    matches = list(re.finditer(pattern, text))
    
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        
        sec_id = matches[i].group('id')
        sec_title = matches[i].group('title').strip()
        
        if any(noise in sec_title.lower() for noise in ['figure', 'table', 'source', 'january', 'february', 'et al']):
            continue
            
        content = text[start:end].strip()
        content = re.sub(r'\s+', ' ', content)
        sections[f"Section {sec_id}: {sec_title}"] = content
        
    return sections

# =========================
# CREATE JSON STRUCTURE
# =========================
def create_json_structure(pdf_path, raw_text):
    # Process text using the robust functions
    title, authors = extract_title_and_authors(raw_text)
    abstract = extract_abstract(raw_text)
    sections_dict = extract_sections(raw_text)
    
    # Create a unified content string for the summarizer model
    if sections_dict:
        content_string = "\n\n".join(list(sections_dict.values()))
    else:
        # Fallback if no sections are found: use cleaned raw text minus abstract
        content_string = clean_text(raw_text)
        if abstract in content_string:
            content_string = content_string.split(abstract, 1)[-1].strip()

    document_id = str(uuid.uuid4())

    paper_json = {
        "document_id": document_id,
        "source_file": os.path.basename(pdf_path),
        "metadata": {
            "title": title,
            "authors": authors,
            "publication_year": None,
            "doi": None,
            "keywords": [],
            "created_at": datetime.utcnow().isoformat()
        },
        "abstract": abstract,
        "sections": sections_dict,  
        "content": content_string   
    }

    return paper_json


# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":
    data_folder = r"data"
    output_dir = "Final_parsed_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading models... This may take a moment.")
    # Load model and tokenizer 
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Models loaded successfully!\n")

    # ----------------------------
    # LOOP THROUGH ALL PDFs
    # ----------------------------
    for file_name in os.listdir(data_folder):
        
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, file_name)
            print(f"Processing: {file_name}")
            
            try:
                # 1. Extract raw text
                raw_text = extract_pdf_text(pdf_path)
                
                # 2. Extract metadata and build structure
                paper_data = create_json_structure(pdf_path, raw_text)

                # 3. Summarize the content
                paper_data['summary'] = summeriser(paper_data['content'], tokenizer, model)
                
                # 4. Extract insights using your helper function
                paper_data['insigth'] = insigth_extraction(paper_data['summary'])
                
                # 5. Save to JSON
                output_path = os.path.join(output_dir, f"{paper_data['document_id']}.json")

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(paper_data, f, indent=4)
                
                print(f"Saved: {output_path}\n")
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}\n")

    print("All PDFs processed successfully ✅")