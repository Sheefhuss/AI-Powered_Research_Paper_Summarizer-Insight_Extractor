import os
import io
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq_file import ask_groq
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j123")
FAISS_PATH     = os.getenv("FAISS_PATH",     "research_papers_faiss")

st.set_page_config(
    page_title="ResearchAI — Paper Intelligence",
    layout="wide",
    page_icon="⚗️",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #020408;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,200,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(120,40,255,0.06) 0%, transparent 50%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%2300c8ff' fill-opacity='0.02'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    min-height: 100vh;
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── Hero Header ── */
.hero-section {
    position: relative;
    text-align: center;
    padding: 3.5rem 2rem 2rem;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 2px;
    background: linear-gradient(90deg, transparent, #00c8ff, #7828ff, transparent);
    animation: scanline 3s ease-in-out infinite;
}
@keyframes scanline {
    0%, 100% { opacity: 0.4; width: 300px; }
    50% { opacity: 1; width: 600px; }
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #00c8ff 40%, #7828ff 80%, #ff6b6b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0;
    animation: fadeSlideDown 0.8s ease forwards;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #00c8ff;
    letter-spacing: 0.15em;
    margin-top: 0.75rem;
    opacity: 0;
    animation: fadeSlideDown 0.8s ease 0.2s forwards;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 100px;
    padding: 6px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #00c8ff;
    margin-top: 1rem;
    letter-spacing: 0.1em;
    opacity: 0;
    animation: fadeSlideDown 0.8s ease 0.4s forwards;
}
.hero-badge::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00c8ff;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.8)} }
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 4px !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: rgba(255,255,255,0.45) !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    transition: all 0.25s ease !important;
    border: 1px solid transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: rgba(255,255,255,0.8) !important;
    background: rgba(255,255,255,0.05) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,200,255,0.15), rgba(120,40,255,0.15)) !important;
    color: #ffffff !important;
    border-color: rgba(0,200,255,0.25) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Cards ── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(20px);
    transition: border-color 0.3s ease, transform 0.3s ease;
    margin-bottom: 1rem;
}
.glass-card:hover {
    border-color: rgba(0,200,255,0.2);
    transform: translateY(-2px);
}

/* ── Search Input ── */
.stTextInput > div > div > input,
.stTextInput > div > div > div > input,
[data-baseweb="input"] input,
[data-baseweb="base-input"] input {
    background: rgba(10,15,25,0.95) !important;
    border: 1px solid rgba(0,200,255,0.2) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    padding: 14px 18px !important;
    transition: all 0.3s ease !important;
    caret-color: #00c8ff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stTextInput > div > div > input:focus,
[data-baseweb="input"] input:focus {
    border-color: rgba(0,200,255,0.6) !important;
    background: rgba(0,200,255,0.06) !important;
    box-shadow: 0 0 0 3px rgba(0,200,255,0.1) !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stTextInput > div > div > input::placeholder,
[data-baseweb="input"] input::placeholder {
    color: rgba(255,255,255,0.3) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.3) !important;
}
[data-baseweb="base-input"] {
    background: rgba(10,15,25,0.95) !important;
    border-radius: 12px !important;
}
.stTextInput label {
    color: rgba(255,255,255,0.6) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    position: relative !important;
    overflow: hidden !important;
    letter-spacing: 0.03em !important;
}
.stButton > button[kind="primary"],
.stButton > button:first-child {
    background: linear-gradient(135deg, #00c8ff, #7828ff) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 20px rgba(0,200,255,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 30px rgba(0,200,255,0.35) !important;
}
.stButton > button:active {
    transform: translateY(-1px) scale(0.99) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.3s ease !important;
}
.stSelectbox > div > div:hover {
    border-color: rgba(0,200,255,0.4) !important;
}
.stSelectbox label {
    color: rgba(255,255,255,0.6) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.85rem !important;
}

/* ── Metrics ── */
.stMetric {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
    transition: all 0.3s ease !important;
}
.stMetric:hover {
    border-color: rgba(0,200,255,0.25) !important;
    background: rgba(0,200,255,0.04) !important;
    transform: translateY(-3px);
}
.stMetric label {
    color: rgba(255,255,255,0.45) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #00c8ff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* ── Expander ── */
details {
    background: rgba(8,12,22,0.98) !important;
    border: 1px solid rgba(0,200,255,0.15) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    margin-bottom: 0.75rem !important;
}
details summary {
    background: rgba(10,16,28,0.99) !important;
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    padding: 14px 18px !important;
    cursor: pointer !important;
    transition: background 0.2s ease !important;
    list-style: none !important;
}
details summary:hover {
    background: rgba(0,200,255,0.08) !important;
}
details[open] summary {
    border-bottom: 1px solid rgba(0,200,255,0.1) !important;
    color: #00c8ff !important;
}
details > div, details > section {
    background: rgba(8,12,22,0.98) !important;
    padding: 1rem 1.25rem !important;
    color: rgba(255,255,255,0.85) !important;
}
.streamlit-expanderHeader,
[data-testid="stExpander"] summary {
    background: rgba(10,16,28,0.99) !important;
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stExpander"] {
    background: rgba(8,12,22,0.98) !important;
    border: 1px solid rgba(0,200,255,0.15) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] * {
    color: rgba(255,255,255,0.85) !important;
    background: transparent !important;
}
[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stExpander"] div {
    background: transparent !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #00c8ff !important;
}

/* ── Alert / Info / Success / Warning ── */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stSuccess {
    background: rgba(0,255,136,0.08) !important;
    border-left: 3px solid #00ff88 !important;
    color: #00ff88 !important;
}
.stInfo {
    background: rgba(0,200,255,0.08) !important;
    border-left: 3px solid #00c8ff !important;
    color: #00c8ff !important;
}
.stWarning {
    background: rgba(255,170,0,0.08) !important;
    border-left: 3px solid #ffaa00 !important;
}
.stError {
    background: rgba(255,60,60,0.08) !important;
    border-left: 3px solid #ff3c3c !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
}
iframe { border-radius: 12px !important; }

/* ── Global text fix ── */
.stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: rgba(255,255,255,0.85) !important;
}
p, li, span, div {
    -webkit-text-fill-color: inherit !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.3), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ── Subheaders ── */
h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: #ffffff !important;
}

/* ── Answer Box ── */
.answer-box {
    background: linear-gradient(135deg, rgba(0,200,255,0.06), rgba(120,40,255,0.06));
    border: 1px solid rgba(0,200,255,0.15);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    position: relative;
    overflow: hidden;
    animation: fadeIn 0.5s ease forwards;
    color: rgba(255,255,255,0.88) !important;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.97rem;
    line-height: 1.75;
}
.answer-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00c8ff, #7828ff);
    border-radius: 3px 0 0 3px;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Model badge ── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 100px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
    animation: fadeIn 0.4s ease forwards;
}
.model-badge.groq {
    background: rgba(249,115,22,0.12);
    border: 1px solid rgba(249,115,22,0.3);
    color: #f97316;
}
.model-badge.gemini {
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.3);
    color: #60a5fa;
}
.model-badge .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
}
.model-badge.groq .dot { background: #f97316; }
.model-badge.gemini .dot { background: #60a5fa; }

/* ── Section Label ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
}

/* ── Paper count badge ── */
.db-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: rgba(0,255,136,0.06);
    border: 1px solid rgba(0,255,136,0.15);
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #00ff88;
    animation: fadeIn 0.5s ease 0.2s both;
}
.db-number {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    color: #00ff88;
}

/* ── Legend pills ── */
.legend-wrap {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.legend-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 100px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    border: 1px solid;
    transition: transform 0.2s ease;
}
.legend-pill:hover { transform: scale(1.05); }
.legend-pill .dot { width: 8px; height: 8px; border-radius: 50%; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-title">Research Intelligence</div>
    <div class="hero-sub">⚗️ AI-POWERED PAPER ANALYSIS &amp; KNOWLEDGE GRAPH</div>
    <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-top:1rem;">
        <div class="hero-badge">SEMANTIC SEARCH</div>
        <div class="hero-badge">KNOWLEDGE GRAPH</div>
        <div class="hero-badge">MULTI-MODEL AI</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["⚗️  Research Paper QA", "🕸️  Knowledge Graph Explorer"])

# ─────────────────────────────────────────────────────────────
# TAB 1 — Research Paper QA
# ─────────────────────────────────────────────────────────────
with tab1:

    @st.cache_resource
    def load_vector_db():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    try:
        vector_db = load_vector_db()
        st.markdown(f"""
        <div class="db-badge">
            <span class="db-number">{vector_db.index.ntotal}</span>
            research papers indexed and ready for semantic search
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Failed to load FAISS database: {e}")
        st.stop()

    st.markdown('<div class="section-label">query interface</div>', unsafe_allow_html=True)

    user_query = st.text_input(
        "Search Query",
        placeholder="✦  Ask anything about the research papers — methods, findings, authors...",
        label_visibility="collapsed"
    )

    col_btn, col_hint = st.columns([1, 4])
    with col_btn:
        search_clicked = st.button("⚡  Search Papers", use_container_width=True)
    with col_hint:
        st.markdown(
            "<p style='color:rgba(255,255,255,0.2); font-size:0.8rem; font-family:JetBrains Mono; padding-top:14px;'>"
            "Powered by FAISS vector search + Groq / Gemini LLM</p>",
            unsafe_allow_html=True
        )

    if search_clicked:
        if not user_query.strip():
            st.warning("⚠️ Please enter a question before searching.")
        else:
            results = vector_db.similarity_search(user_query, k=3)
            content = ""
            for idx, doc in enumerate(results, 1):
                title = doc.metadata.get("title", f"Paper {idx}")
                content += f"\nPaper Title: {title}\nPaper Content:\n{doc.page_content}\n"

            with st.spinner("⚗️  Analyzing papers and synthesizing insights..."):
                response = ask_groq(content, user_query)

            used_gemini   = response.startswith("[Answered by Gemini]")
            clean_response = response.replace("[Answered by Gemini]", "").strip()

            answer = ""
            paper_titles = []
            if "Research Paper:" in clean_response:
                parts = clean_response.split("Research Paper:")
                answer = parts[0].replace("Answer:", "").strip()
                paper_titles = [p.strip() for p in parts[1].strip().split(",")]
            else:
                answer = clean_response.strip()

            # Model badge
            if used_gemini:
                st.markdown('<div class="model-badge gemini"><span class="dot"></span>ANSWERED BY GEMINI 2.0 FLASH</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="model-badge groq"><span class="dot"></span>ANSWERED BY GROQ — LLAMA 3.3 70B</div>', unsafe_allow_html=True)

            # Answer
            st.markdown('<div class="section-label">ai generated insight</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            # Source papers
            if paper_titles and "none" not in [p.lower() for p in paper_titles]:
                st.markdown("")
                st.markdown('<div class="section-label">source papers</div>', unsafe_allow_html=True)
                for doc in results:
                    title = doc.metadata.get("title", "")
                    for p in paper_titles:
                        if title.lower() == p.lower():
                            with st.expander(f"📄  {title}", expanded=True):
                                st.markdown(
                                    f"<div style='font-family:Space Grotesk; color:rgba(255,255,255,0.75); "
                                    f"line-height:1.7; font-size:0.92rem;'>{doc.page_content}</div>",
                                    unsafe_allow_html=True
                                )

# ─────────────────────────────────────────────────────────────
# TAB 2 — Knowledge Graph Explorer
# ─────────────────────────────────────────────────────────────
with tab2:

    @st.cache_resource
    def init_neo4j_driver():
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        driver = init_neo4j_driver()
        with driver.session() as _s:
            _s.run("RETURN 1")
    except Exception as e:
        st.error(f"❌ Cannot connect to Neo4j: {e}")
        st.stop()

    @st.cache_data
    def get_domains():
        query = "MATCH (d:Domain) RETURN d.name AS domain"
        try:
            with driver.session() as session:
                result  = session.run(query)
                domains = [r["domain"] for r in result if r["domain"]]
            normalized = {}
            for d in domains:
                normalized[d.lower()] = d.title()
            return sorted(normalized.values())
        except Exception as e:
            st.error(f"❌ Error fetching domains: {e}")
            return []

    domains = get_domains()
    if not domains:
        st.warning("⚠️ No domains found. Run build_graph.py first.")
        st.stop()

    st.markdown('<div class="section-label">domain selector</div>', unsafe_allow_html=True)

    col_domain, col_refresh = st.columns([4, 1])
    with col_domain:
        domain = st.selectbox(
            "Research Domain",
            options=["— Select a research domain —"] + domains,
            index=0,
            label_visibility="collapsed"
        )
    with col_refresh:
        if st.button("↺  Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    def get_graph_data(domain):
        query = """
        MATCH (p:Paper)-[:BELONGS_TO]->(d:Domain)
        WHERE toLower(d.name) = toLower($domain)
        OPTIONAL MATCH (p)<-[:WROTE]-(a:Author)
        OPTIONAL MATCH (p)-[:USES]->(m:Method)
        RETURN p.title AS paper, a.name AS author, m.name AS method, d.name AS domain
        """
        with driver.session() as session:
            result = session.run(query, domain=domain)
            return [r.data() for r in result]

    def draw_graph(data):
        net = Network(height="660px", width="100%", bgcolor="#020408", font_color="white", notebook=False)
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": { "iterations": 150 },
            "barnesHut": {
              "gravitationalConstant": -9000,
              "centralGravity": 0.3,
              "springLength": 160,
              "springConstant": 0.04,
              "damping": 0.09
            }
          },
          "nodes": {
            "font": { "size": 11, "face": "Space Grotesk" },
            "borderWidth": 2,
            "shadow": { "enabled": true, "size": 12 }
          },
          "edges": {
            "smooth": { "type": "curvedCW", "roundness": 0.2 },
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.4 } },
            "width": 1.2,
            "shadow": { "enabled": true }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true
          }
        }
        """)

        added_nodes = set()
        for row in data:
            paper  = row.get("paper")
            author = row.get("author")
            method = row.get("method")
            dom    = row.get("domain")

            if paper and paper not in added_nodes:
                net.add_node(paper, label=paper[:28]+"…" if len(paper)>28 else paper,
                             color={"background":"#F4A261","border":"#e76f51","highlight":{"background":"#ffb347","border":"#ff8c00"}},
                             size=22, title=f"📄 {paper}", shape="dot")
                added_nodes.add(paper)
            if author and author not in added_nodes:
                net.add_node(author, label=author,
                             color={"background":"#00c8ff","border":"#0096c7","highlight":{"background":"#48cae4","border":"#0077b6"}},
                             size=15, title=f"👤 {author}", shape="dot")
                added_nodes.add(author)
            if method and method not in added_nodes:
                net.add_node(method, label=method,
                             color={"background":"#52b788","border":"#2d6a4f","highlight":{"background":"#74c69d","border":"#1b4332"}},
                             size=15, title=f"⚙️ {method}", shape="dot")
                added_nodes.add(method)
            if dom and dom not in added_nodes:
                net.add_node(dom, label=dom,
                             color={"background":"#c77dff","border":"#7b2d8b","highlight":{"background":"#e0aaff","border":"#9d4edd"}},
                             size=30, title=f"🔬 {dom}", shape="dot")
                added_nodes.add(dom)

            if author and paper:
                net.add_edge(author, paper, title="WROTE", color={"color":"rgba(0,200,255,0.35)","highlight":"#00c8ff"})
            if paper and method:
                net.add_edge(paper, method, title="USES", color={"color":"rgba(82,183,136,0.35)","highlight":"#52b788"})
            if paper and dom:
                net.add_edge(paper, dom, title="BELONGS TO", color={"color":"rgba(199,125,255,0.35)","highlight":"#c77dff"})

        net.save_graph("graph.html")
        with open("graph.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=680)

    if domain and domain != "— Select a research domain —":

        with st.spinner("🕸️  Loading knowledge graph..."):
            data = get_graph_data(domain)

        if not data:
            st.warning("No papers found for this domain.")
        else:
            df = pd.DataFrame(data)
            papers  = df["paper"].nunique()
            authors = df["author"].dropna().nunique()
            methods = df["method"].dropna().nunique()

            st.markdown('<div class="section-label">domain overview</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Papers",  papers)
            col2.metric("Authors", authors)
            col3.metric("Methods", methods)

            st.markdown("")
            st.markdown("""
            <div class="legend-wrap">
                <div class="legend-pill" style="background:rgba(244,162,97,0.1);border-color:rgba(244,162,97,0.3);color:#f4a261;">
                    <span class="dot" style="background:#f4a261;"></span>Paper
                </div>
                <div class="legend-pill" style="background:rgba(0,200,255,0.1);border-color:rgba(0,200,255,0.3);color:#00c8ff;">
                    <span class="dot" style="background:#00c8ff;"></span>Author
                </div>
                <div class="legend-pill" style="background:rgba(82,183,136,0.1);border-color:rgba(82,183,136,0.3);color:#52b788;">
                    <span class="dot" style="background:#52b788;"></span>Method
                </div>
                <div class="legend-pill" style="background:rgba(199,125,255,0.1);border-color:rgba(199,125,255,0.3);color:#c77dff;">
                    <span class="dot" style="background:#c77dff;"></span>Domain
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">interactive graph — drag · zoom · hover</div>', unsafe_allow_html=True)
            draw_graph(data)

            st.markdown("")
            with st.expander("📊  View Data Table & Export"):
                st.dataframe(df, use_container_width=True)
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False)
                excel_buffer.seek(0)
                st.download_button(
                    label="⬇️  Export as Excel",
                    data=excel_buffer,
                    file_name=f"{domain}_research_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; opacity:0.5;">
            <div style="font-size:3rem; margin-bottom:1rem;">🕸️</div>
            <div style="font-family:Syne,sans-serif; font-size:1.2rem; color:white; margin-bottom:0.5rem;">
                Select a domain to explore the knowledge graph
            </div>
            <div style="font-family:JetBrains Mono,monospace; font-size:0.75rem; color:rgba(255,255,255,0.4); letter-spacing:0.1em;">
                PAPERS · AUTHORS · METHODS · RELATIONSHIPS
            </div>
        </div>
        """, unsafe_allow_html=True)