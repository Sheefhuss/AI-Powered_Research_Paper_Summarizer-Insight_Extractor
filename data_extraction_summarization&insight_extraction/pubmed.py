import requests
import xml.etree.ElementTree as ET
import pandas as pd
import json
import time
from helper_function import *

search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

queries = [
    "precision oncology",
    "tumor microenvironment",
    "CAR T cell therapy",
    "cancer genomics",
    "breast cancer biomarkers",
    "AI in cancer diagnosis",
    "machine learning",
    "large language model",
    "generative AI",
    "retrieval augmented generation",
    "semantic search",
    "vector database",
    "sentence transformer"
]

file_name = "pubmed_multiple_queries1.json"

all_papers = []
seen_pmids = set()

try:
    for query in queries:
        print(f"\n🔎 Searching for: {query}")

        try:
            # ---------- SEARCH ----------
            params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": 10
            }

            response = requests.get(search_url, params=params, timeout=20)
            response.raise_for_status()

            pmids = response.json()["esearchresult"]["idlist"]
            if not pmids:
                continue

            print("PMIDs:", pmids)
            time.sleep(1)

            # ---------- FETCH ----------
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml"
            }

            xml_data = requests.get(fetch_url, params=fetch_params, timeout=20).text
            root = ET.fromstring(xml_data)

            # ---------- PARSE ----------
            for article in root.findall(".//PubmedArticle"):
                try:
                    pmid = article.findtext(".//PMID")

                    if not pmid or pmid in seen_pmids:
                        continue
                    seen_pmids.add(pmid)

                    abstract_sections = []
                    abstract_node = article.find(".//Abstract")

                    if abstract_node is not None:
                        for abs_text in abstract_node.findall(".//AbstractText"):
                            label = abs_text.attrib.get("Label")
                            text = "".join(abs_text.itertext()).strip()
                            if text:
                                abstract_sections.append(
                                    f"{label}: {text}" if label else text
                                )

                    full_abstract = "\n".join(abstract_sections) if abstract_sections else None

                    data = {
                        "query": query,
                        "pmid": pmid,
                        "title": article.findtext(".//ArticleTitle"),
                        "journal": article.findtext(".//Journal/Title"),
                        "doi": article.findtext(".//ELocationID[@EIdType='doi']"),
                        "abstract": full_abstract,
                        "keywords": [k.text for k in article.findall(".//Keyword") if k.text],
                        "authors": [
                            f"{a.findtext('ForeName', '')} {a.findtext('LastName', '')}".strip()
                            for a in article.findall(".//Author")
                        ],
                        "insight": insigth_extraction(full_abstract) if full_abstract else None
                    }

                    all_papers.append(data)

                except Exception as article_error:
                    print(f"⚠️ Error parsing article PMID {pmid}: {article_error}")

            time.sleep(1)

        except Exception as query_error:
            print(f"❌ Error processing query '{query}': {query_error}")

except Exception as fatal_error:
    print("🚨 Fatal error occurred:", fatal_error)

finally:
    # ---------- ALWAYS SAVE ----------
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=4, ensure_ascii=False)

    print(f"\n💾 Data saved to {file_name}")
    print(f"📊 Total papers collected: {len(all_papers)}")