# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from  gemini_file import *

# # load embedding model 
# embeddings = HuggingFaceEmbeddings(
#     model_name = "sentence-transformers/all-MiniLM-L6-v2"
# )


# # load FAISS vector store 

# vector_db = FAISS.load_local(
#     "research_papers_faiss",
#     embeddings,
#     allow_dangerous_deserialization= True
# )

# print(f"vector loaded: {vector_db.index.ntotal}")

# # user query 
# # user_query ="Which papers discuss privacy in ML"
# user_query="which paper use entropy-based evaluation"

# # similarity search , Keyword search, hybrid search

# # retrival 
# top_k =3
# results = vector_db.similarity_search(user_query, k=top_k)

# # display results 
# print("\n Top Relevent Documents:\n")

# content=""
# for idx, doc in enumerate(results, 1):
    
#     print(f"{idx, {doc.page_content}}")
#     content+=", "+ doc.page_content



# print("-------- Gemini Response-------------")
# ask_gemini(content, user_query)




# # https://chatgpt.com/c/69a86e7d-3bb8-8324-9eaf-c5807d399382

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from gemini_file import ask_gemini

# Title
# color:#00FFFF;
st.markdown(
    "<h1 style='text-align:center; '>AI-Powered Research Paper Summarizer & Insight Extractor</h1>",
    unsafe_allow_html=True
)

# Small description
st.markdown(
    "<p style='text-align:center; color:white;'>Ask questions and get AI-powered insights from research papers</p>",
    unsafe_allow_html=True
)

# Custom CSS for input box
st.markdown("""
<style>
.stTextInput input {
    background-color: #F5F5F5;
    color: #000000;
    border-radius: 10px;
    border: 2px solid #00FFFF;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vector_db():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.load_local(
        "research_papers_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_db


vector_db = load_vector_db()

# Show total papers
st.write("📚 Total Papers in Database:", vector_db.index.ntotal)

# User input
user_query = st.text_input("🔎 Ask a question about research papers:")

# Search button
if st.button("Search"):

    results = vector_db.similarity_search(user_query, k=3)

    content = ""

    for idx, doc in enumerate(results, 1):

        title = doc.metadata.get("title", f"Paper {idx}")

        content += f"""
        Paper Title: {title}

        Paper Content:
        {doc.page_content}

        """

    print("whole content",content )
    # Call Gemini
    with st.spinner("🤖 Analyzing research papers and generating insights..."):
        response = ask_gemini(content, user_query)
    print("gemini response", response)
    # -------------------------
    # Extract Answer and Paper
    # -------------------------

    answer = ""
    paper_titles = []

    if "Research Paper:" in response:
        parts = response.split("Research Paper:")
        
        answer = parts[0].replace("Answer:", "").strip()

        papers_text = parts[1].strip()

        # Split multiple papers by comma
        paper_titles = [p.strip() for p in papers_text.split(",")]

    else:
        answer = response.strip()

    st.subheader("🤖 AI Generated Insight")
    st.write(answer)


    if paper_titles and "none" not in [p.lower() for p in paper_titles]:

        st.subheader("📄 Relevant Research Papers")

        for doc in results:

            title = doc.metadata.get("title", "")

            for p in paper_titles:

                if title.lower() == p.lower():

                    with st.expander(f"📄 {title}"):

                        st.write(doc.page_content)

    else:
        st.warning("No relevant research paper found.")