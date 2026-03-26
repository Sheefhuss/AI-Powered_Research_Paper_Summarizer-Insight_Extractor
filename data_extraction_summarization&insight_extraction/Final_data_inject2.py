import requests
import feedparser
import pandas as pd
import time
import json
from helper_function import *

def fetch_arxiv_papers(query, max_results=50, start=0):
    url = "http://export.arxiv.org/api/query"
    
    # Add 'all:' dynamically for the API request only
    api_query = f"all:{query}"
    
    params = {
        "search_query": api_query,
        "start": start,
        "max_results": max_results
    }

    response = requests.get(url, params=params)
    feed = feedparser.parse(response.text)

    papers = []
    for entry in feed.entries:
        papers.append({
            "source": "arxiv",
            "search_query": query,  # Saves the clean query without 'all:'
            "paper_id": entry.id,
            "title": entry.title.strip(),
            "authors": ", ".join([a.name for a in entry.authors]),
            "abstract": entry.summary.strip(),
            "published": entry.published,
            "categories": ", ".join([tag.term for tag in entry.tags]),
            "pdf_url": next(
                (link.href for link in entry.links if link.type == "application/pdf"),
                None
            ),
            "insight": insigth_extraction(entry.summary.strip()) if entry.summary else None
        })

    time.sleep(3) 
    return papers

# ----------------------------
# SEARCH QUERIES (Cleaned)
# ----------------------------
queries = [
    '"machine learning"',
    '"large language model"',
    '"generative AI"',
    '"retrieval augmented generation" OR RAG',
    '"semantic search"',
    '"vector database"',
    '"sentence transformer"'
]

all_papers = []

# ----------------------------
# LOOP THROUGH QUERIES
# ----------------------------
for query in queries:
    print(f"Fetching papers for: {query}")
    papers = fetch_arxiv_papers(query, max_results=20)
    all_papers.extend(papers)

# ----------------------------
# REMOVE DUPLICATES
# ----------------------------
unique_papers = {paper["paper_id"]: paper for paper in all_papers if paper.get("paper_id")}
final_papers = list(unique_papers.values())

# ----------------------------
# SAVE TO JSON
# ----------------------------
with open("arxiv_papers_vb.json", "w", encoding="utf-8") as f:
    json.dump(final_papers, f, indent=4, ensure_ascii=False)

print("Saved successfully to arxiv_papers.json ✅")