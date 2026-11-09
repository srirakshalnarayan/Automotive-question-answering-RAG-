
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import faiss
import requests
import json

from pathlib import Path

# ------------------------------------------------
# 1. PROJECT CONFIGURATION
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_FILE = BASE_DIR / "vector_store" / "automotive_metadata.csv"
INDEX_FILE = BASE_DIR / "vector_store" / "automotive_faiss.index"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3.5:4b"

# ------------------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="Automotive AI Assistant",
    page_icon="🚗",
    layout="wide"
)

# ------------------------------------------------
# 3. CUSTOM UI STYLING
# ------------------------------------------------

st.markdown("""
<style>

/* Main application */

.stApp {
    background: #080f1d;
    color: #e5e7eb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1300px;
}

/* Headings */

h2, h3 {
    color: #f1f5f9 !important;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: #101b2e;
    border-right: 1px solid #263b56;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

.sidebar-card {
    background: linear-gradient(145deg, #172942, #122138);
    border: 1px solid #2b4565;
    border-radius: 13px;
    padding: 17px;
    margin: 12px 0;
}

.sidebar-card p {
    color: #cbd5e1;
    font-size: 13px;
    line-height: 1.7;
    margin: 8px 0;
}

.developer-card {
    background: #162c48;
    border: 1px solid #3b6591;
    border-radius: 12px;
    padding: 15px;
    margin-top: 20px;
    text-align: center;
}

.developer-card .name {
    color: #ffffff;
    font-size: 17px;
    font-weight: 750;
}

.developer-card .role {
    color: #7db8ff;
    font-size: 11px;
    letter-spacing: 1.5px;
    margin-top: 5px;
}

/* Question input */

div[data-testid="stTextInput"] input {
    background: #17263d;
    color: #ffffff;
    border: 1px solid #3a5270;
    border-radius: 10px;
    padding: 14px;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 1px #3b82f6;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
    color: white;
    border: none;
    box-shadow: 0 0 16px rgba(59, 130, 246, 0.35);
}

/* Answer heading */

.answer-heading {
    color: #60a5fa;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 15px;
}

/* Retrieved source expanders */

[data-testid="stExpander"] {
    background: #111f34;
    border: 1px solid #2b4565;
    border-radius: 11px;
    margin-bottom: 10px;
}

[data-testid="stExpander"] summary {
    color: #e2e8f0;
}

/* Download button */

.stDownloadButton > button {
    background: #172d49;
    color: #93c5fd;
    border: 1px solid #3b82f6;
    border-radius: 9px;
}

/* Divider */

hr {
    border-color: #293d56;
}

.stCaption {
    color: #94a3b8;
}

/* Footer */

.footer {
    text-align: center;
    padding: 25px 10px 10px 10px;
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.9;
}

.footer-name {
    color: #60a5fa;
    font-size: 15px;
    font-weight: 750;
    letter-spacing: 1px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# 4. AUTOMOTIVE HEADER WITH ANIMATED CAR
# ------------------------------------------------

components.html("""
<!DOCTYPE html>
<html>
<head>
<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: transparent;
    font-family: Arial, sans-serif;
}

.hero {
    background: linear-gradient(120deg, #101d35, #142e50, #10213b);
    padding: 35px 38px 25px 38px;
    border-radius: 18px;
    border: 1px solid #29496d;
    box-shadow: 0 8px 35px rgba(0, 0, 0, 0.25);
    overflow: hidden;
}

.label {
    color: #60a5fa;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
}

h1 {
    color: #ffffff;
    font-size: 36px;
    margin: 15px 0 10px 0;
}

.description {
    color: #c5d5e8;
    font-size: 15px;
    line-height: 1.8;
    max-width: 850px;
}

.road {
    position: relative;
    height: 48px;
    margin-top: 24px;
    background: #0c1728;
    border-top: 2px solid #426080;
    border-bottom: 2px solid #426080;
    overflow: hidden;
    border-radius: 6px;
}

.road::after {
    content: "";
    position: absolute;
    top: 22px;
    left: 0;
    width: 200%;
    border-top: 2px dashed #9bb8d5;
    animation: roadMove 2s linear infinite;
}

.car {
    position: absolute;
    top: 2px;
    left: -10%;
    font-size: 30px;
    z-index: 2;
    transform: scaleX(-1);
    animation: drive 9s linear infinite;
    filter: drop-shadow(0 0 8px #3b82f6);
}

@keyframes drive {
    0% { left: -10%; }
    100% { left: 110%; }
}

@keyframes roadMove {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}

@media (max-width: 650px) {
    .hero {
        padding: 25px 20px;
    }

    h1 {
        font-size: 27px;
    }

    .description {
        font-size: 13px;
    }
}

</style>
</head>

<body>

<div class="hero">

    <div class="label">
        ENGINEERED WITH NLP • POWERED BY AI
    </div>

    <h1>🚗 Automotive AI Assistant</h1>

    <div class="description">
        Your intelligent automotive knowledge companion.
        Explore automotive questions, retrieve relevant information
        and generate contextual answers using Natural Language
        Processing and Retrieval-Augmented Generation.
    </div>

    <div class="road">
        <div class="car">🏎️</div>
    </div>

</div>

</body>
</html>
""", height=275, scrolling=False)

# ------------------------------------------------
# 5. SIDEBAR
# ------------------------------------------------

with st.sidebar:

    st.markdown("## 🚘 Project Overview")

    st.markdown("""
<div class="sidebar-card">
<p><b>ABOUT THE APPLICATION</b></p>
<p>
An intelligent automotive question-answering system
that retrieves relevant information from an automotive
dataset and generates contextual responses using an
AI language model.
</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### ⚙️ Technology Stack")

    st.markdown("""
<div class="sidebar-card">
<p>📚 <b>Dataset:</b> Automotive Q&A</p>
<p>🧠 <b>Embeddings:</b> MiniLM</p>
<p>🔎 <b>Retrieval:</b> FAISS</p>
<p>🎯 <b>Reranking:</b> Cross-Encoder</p>
<p>🤖 <b>LLM:</b> Qwen3.5 4B</p>
<p>⚡ <b>Runtime:</b> Ollama</p>
<p>🖥️ <b>Interface:</b> Streamlit</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="developer-card">
<div style="font-size:25px;">👩‍💻</div>
<div class="name">Sriraksha L</div>
<div class="role">PROJECT DEVELOPER</div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# 6. LOAD COMPONENTS
# ------------------------------------------------

@st.cache_resource
def load_components():

    from sentence_transformers import SentenceTransformer, CrossEncoder

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Automotive metadata file not found."
        )

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            "FAISS index file not found."
        )

    df = pd.read_csv(METADATA_FILE)

    index = faiss.read_index(str(INDEX_FILE))

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return df, index, embedding_model, reranker

# ------------------------------------------------
# 7. RETRIEVAL
# ------------------------------------------------

def retrieve_documents(query, df, index, embedding_model):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        20
    )

    documents = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        row = df.iloc[idx]

        documents.append({
            "question": row["clean_question"],
            "answer": row["clean_answer"],
            "faiss_score": float(score)
        })

    return documents

# ------------------------------------------------
# 8. RERANKING
# ------------------------------------------------

def rerank_documents(query, documents, reranker):

    if not documents:
        return []

    pairs = [
        [query, doc["question"]]
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    for doc, score in zip(documents, scores):

        doc["reranker_score"] = float(score)

    documents.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return documents[:3]

# ------------------------------------------------
# 9. CONTEXT PREPARATION
# ------------------------------------------------

def prepare_context(documents):

    context = ""

    for i, doc in enumerate(documents, 1):

        context += (
            f"\nSOURCE {i}\n"
            f"Question: {doc['question']}\n"
            f"Answer: {doc['answer']}\n"
            f"{'-' * 50}\n"
        )

    return context.strip()

# ------------------------------------------------
# 10. QWEN ANSWER GENERATION - STREAMING
# ------------------------------------------------

def generate_answer(query, context):

    prompt = f"""
You are an automotive question-answering assistant.

Answer the user's question using the retrieved context.

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:
1. Use the retrieved information as your primary source.
2. Do not invent facts.
3. If the context is insufficient, clearly say so.
4. Give a concise and understandable answer.
5. Keep the answer relevant to the question.

FINAL ANSWER:
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "think": False,
        "stream": True,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.2,
            "num_predict": 240,
            "num_ctx": 2048
        }
    }

    with requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=(10, 300)
    ) as response:

        response.raise_for_status()

        for line in response.iter_lines():

            if not line:
                continue

            data = json.loads(line)

            content = data.get(
                "message", {}
            ).get("content", "")

            if content:
                yield content

            if data.get("done"):
                break

# ------------------------------------------------
# 11. QUESTION INPUT
# ------------------------------------------------

st.markdown("""
<div style="background:#101e32;
            border:1px solid #263f5e;
            border-radius:15px;
            padding:22px;
            margin-top:20px;
            margin-bottom:15px;">

<div style="color:#60a5fa;font-size:12px;
            font-weight:800;letter-spacing:2px;">
YOUR AUTOMOTIVE KNOWLEDGE HUB
</div>

<div style="color:white;font-size:23px;
            font-weight:750;margin-top:7px;">
💬 What would you like to know?
</div>

<div style="color:#a9bad0;font-size:13px;margin-top:5px;">
Ask a question about automobiles and let AI find the answer.
</div>

</div>
""", unsafe_allow_html=True)

query = st.text_input(
    "Enter your question",
    placeholder="e.g. What causes a car engine to overheat?",
    label_visibility="collapsed"
)

search_button = st.button(
    "⚡ Generate Answer",
    type="primary",
    use_container_width=True
)

# ------------------------------------------------
# 12. COMPLETE RAG EXECUTION
# ------------------------------------------------

if search_button:

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        try:

            # Load NLP components
            with st.spinner("Loading NLP components..."):

                df, index, embedding_model, reranker = (
                    load_components()
                )

            # Retrieve documents
            with st.spinner("Retrieving relevant information..."):

                documents = retrieve_documents(
                    query,
                    df,
                    index,
                    embedding_model
                )

            if not documents:

                st.warning("No matching records found.")

            else:

                # Rerank documents
                with st.spinner("Reranking retrieved information..."):

                    documents = rerank_documents(
                        query,
                        documents,
                        reranker
                    )

                # Prepare context
                context = prepare_context(documents)

                # Display answer section
                st.divider()

                st.markdown(
                    '<div class="answer-heading">🤖 AI Generated Answer</div>',
                    unsafe_allow_html=True
                )

                # Stream answer directly to the UI
                answer = st.write_stream(
                    generate_answer(query, context)
                )

                if answer:

                    st.success(
                        "Answer generated successfully!"
                    )

                # Display retrieved sources
                st.divider()

                st.markdown("### 📚 Retrieved Sources")

                for i, doc in enumerate(documents, 1):

                    with st.expander(
                        f"🔹 Source {i}: {doc['question']}"
                    ):

                        st.write("**Retrieved Answer:**")
                        st.write(doc["answer"])

                        st.caption(
                            f"FAISS Score: "
                            f"{doc['faiss_score']:.4f}"
                        )

                        st.caption(
                            f"Reranker Score: "
                            f"{doc['reranker_score']:.4f}"
                        )

                # Display context
                with st.expander("🔍 View Retrieved Context"):

                    st.text(context)

                # Download answer
                st.download_button(
                    "⬇️ Download Generated Answer",
                    data=answer or "",
                    file_name="automotive_answer.txt",
                    mime="text/plain"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to Ollama. "
                "Please make sure Ollama is running."
            )

        except requests.exceptions.Timeout:

            st.error(
                "Qwen is taking too long to respond. "
                "Please try a shorter question."
            )

        except Exception as error:

            st.error(
                f"Application error: {error}"
            )

# ------------------------------------------------
# 13. FOOTER
# ------------------------------------------------

st.divider()

st.markdown("""
<div class="footer">
<div style="font-size:24px;">🏎️</div>
<div style="color:#cbd5e1;font-weight:700;">
AUTOMOTIVE QUESTION UNDERSTANDING AND ANSWER GENERATION
</div>
<div>
Using Natural Language Processing and
Retrieval-Augmented Generation
</div>
<br>
<div class="footer-name">Developed by Sriraksha L</div>
<div>Automotive AI Assistant • 2026</div>
</div>
""", unsafe_allow_html=True)
