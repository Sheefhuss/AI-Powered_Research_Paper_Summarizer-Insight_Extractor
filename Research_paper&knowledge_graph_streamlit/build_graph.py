
import os
import json
import time
import re
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from neo4j import GraphDatabase
from groq import Groq

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("📁 Loaded .env file")
except ImportError:
    pass  

# ─────────────────────────────────────────────────────────────
# CONFIG — all secrets come from environment variables
# ─────────────────────────────────────────────────────────────
FAISS_PATH      = os.getenv("FAISS_PATH", "research_papers_faiss")
NEO4J_URI       = os.getenv("NEO4J_URI",  "bolt://127.0.0.1:7687")
NEO4J_USER      = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD  = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")

GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE", "graph_progress.json")
CONTENT_LIMIT   = 1500   
RETRY_LIMIT     = 5    
REQUEST_DELAY   = 1.5    
RATE_LIMIT_WAIT = 60    

_missing = [k for k, v in {
    "NEO4J_PASSWORD": NEO4J_PASSWORD,
    "GROQ_API_KEY": GROQ_API_KEY
}.items() if not v]

if _missing:
    print(f"\n❌ Missing required environment variables: {', '.join(_missing)}")
    print("   Create a .env file with these values — see .env.example\n")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# LOAD FAISS
# ─────────────────────────────────────────────────────────────
print("📦 Loading FAISS vector store...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)
all_docs = list(vector_db.docstore._dict.values())
print(f"✅ Loaded {len(all_docs)} documents from FAISS\n")

# ─────────────────────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)

def extract_entities(title: str, content: str) -> dict:
    """
    Calls Groq LLM to extract structured entities from a paper.
    Returns a dict with keys: title, authors, methods, domain.
    """
    prompt = f"""Extract structured information from this research paper.
Return ONLY a valid JSON object with no extra text, no markdown, no code blocks.

Paper Title: {title}
Paper Content: {content[:CONTENT_LIMIT]}

Return exactly this JSON structure:
{{
  "title": "the paper title",
  "authors": ["Author Name One", "Author Name Two"],
  "methods": ["Method or Technique One", "Method or Technique Two"],
  "domain": "single domain like Machine Learning or Computer Vision or NLP"
}}

Rules:
- authors: list of person names found in the content (empty list [] if none found)
- methods: list of techniques, algorithms, or models used (empty list [] if none found)
- domain: exactly one domain as a short phrase
- Return ONLY the JSON object, nothing else"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512
    )

    raw = response.choices[0].message.content.strip()

    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    return json.loads(raw)


# ─────────────────────────────────────────────────────────────
# NEO4J WRITER
# ─────────────────────────────────────────────────────────────
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def write_to_neo4j(data: dict):
    """
    MERGEs all nodes and relationships into Neo4j.
    MERGE prevents duplicate nodes if run multiple times.
    """
    title   = data.get("title", "").strip()
    domain  = data.get("domain", "Unknown").strip()
    authors = [a.strip() for a in data.get("authors", []) if a.strip()]
    methods = [m.strip() for m in data.get("methods", []) if m.strip()]

    with driver.session() as session:
        # Paper + Domain nodes
        session.run("MERGE (p:Paper {title: $title})", title=title)
        session.run("MERGE (d:Domain {name: $domain})", domain=domain)

        # Paper → Domain
        session.run("""
            MATCH (p:Paper {title: $title}), (d:Domain {name: $domain})
            MERGE (p)-[:BELONGS_TO]->(d)
        """, title=title, domain=domain)

        # Author nodes + Author → Paper
        for author in authors:
            session.run("MERGE (a:Author {name: $name})", name=author)
            session.run("""
                MATCH (a:Author {name: $name}), (p:Paper {title: $title})
                MERGE (a)-[:WROTE]->(p)
            """, name=author, title=title)

        # Method nodes + Paper → Method
        for method in methods:
            session.run("MERGE (m:Method {name: $name})", name=method)
            session.run("""
                MATCH (p:Paper {title: $title}), (m:Method {name: $name})
                MERGE (p)-[:USES]->(m)
            """, title=title, name=method)


def test_neo4j_connection():
    """Verify Neo4j is reachable before starting the main loop."""
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        print("✅ Neo4j connection OK\n")
        return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("   Make sure Neo4j is running and credentials are correct.\n")
        return False


# ─────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────
def load_checkpoint() -> set:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        print(f"🔖 Resuming — {len(data)} papers already processed\n")
        return set(data)
    return set()


def save_checkpoint(seen: set):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(seen), f, indent=2)


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def main():
    if not test_neo4j_connection():
        return

    seen_titles = load_checkpoint()

    unique_docs = {}
    for doc in all_docs:
        title = doc.metadata.get("title", "").strip()
        if title and title not in unique_docs:
            unique_docs[title] = doc

    total     = len(unique_docs)
    processed = 0
    skipped   = 0
    failed    = 0

    print(f"📄 Total unique papers : {total}")
    print(f"⏭️  Already done       : {len(seen_titles)}")
    print(f"🔄 Remaining           : {total - len(seen_titles)}")
    print(f"🤖 Model               : {GROQ_MODEL}\n")
    print("=" * 60)

    for i, (title, doc) in enumerate(unique_docs.items(), 1):
        if title in seen_titles:
            skipped += 1
            continue

        short_title = title[:65] + "..." if len(title) > 65 else title
        print(f"[{i}/{total}] {short_title}")

        success      = False
        wait_seconds = RATE_LIMIT_WAIT

        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                data = extract_entities(title, doc.page_content)

                data["title"] = title

                write_to_neo4j(data)

                seen_titles.add(title)
                save_checkpoint(seen_titles)

                print(f"  ✓ Domain: {data['domain']}")
                if data.get("authors"):
                    print(f"    Authors : {', '.join(data['authors'][:3])}"
                          + (" ..." if len(data["authors"]) > 3 else ""))
                if data.get("methods"):
                    print(f"    Methods : {', '.join(data['methods'][:3])}"
                          + (" ..." if len(data["methods"]) > 3 else ""))

                processed += 1
                success = True

                #requests to stay within rate limits
                time.sleep(REQUEST_DELAY)
                break

            except json.JSONDecodeError as e:
                print(f"  ⚠️  Attempt {attempt}/{RETRY_LIMIT} — JSON parse error: {e}")
                if attempt < RETRY_LIMIT:
                    time.sleep(3)

            except Exception as e:
                error_msg = str(e)

                # Rate limit hit
                if "429" in error_msg or "rate_limit" in error_msg.lower():
                    print(f"  🚦 Attempt {attempt}/{RETRY_LIMIT} — Rate limit hit. "
                          f"Waiting {wait_seconds}s before retry...")
                    time.sleep(wait_seconds)
                    wait_seconds = min(wait_seconds * 2, 300)  

                elif "model_decommissioned" in error_msg:
                    print("  ❌ Model decommissioned — update GROQ_MODEL in .env")
                    driver.close()
                    sys.exit(1)

                else:
                    print(f"  ⚠️  Attempt {attempt}/{RETRY_LIMIT} — {error_msg[:120]}")
                    if attempt < RETRY_LIMIT:
                        time.sleep(5)

        if not success:
            print(f"  ✗ Skipping after {RETRY_LIMIT} failed attempts")
            failed += 1

    driver.close()
    print("\n" + "=" * 60)
    print("✅ DONE!")
    print(f"   Newly processed : {processed}")
    print(f"   Already done    : {skipped}")
    print(f"   Failed          : {failed}")
    print(f"   Total in graph  : {len(seen_titles)}")
    print("\nVerify in Neo4j Browser:")
    print("  MATCH (n) RETURN n LIMIT 50")
    print("  MATCH (d:Domain) RETURN d.name, count{(p:Paper)-[:BELONGS_TO]->(d)} AS papers ORDER BY papers DESC")


if __name__ == "__main__":
    main()