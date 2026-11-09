
import pandas as pd
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

# ------------------------------------------------
# 1. LOAD PROJECT FILES
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

metadata_file = BASE_DIR / "vector_store" / "automotive_metadata.csv"
index_file = BASE_DIR / "vector_store" / "automotive_faiss.index"

df = pd.read_csv(metadata_file)
index = faiss.read_index(str(index_file))

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ------------------------------------------------
# 2. RETRIEVAL
# ------------------------------------------------

def retrieve_documents(query):

    embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    embedding = np.asarray(embedding, dtype="float32")

    scores, indices = index.search(embedding, 20)

    documents = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        row = df.iloc[idx]

        documents.append({
            "question": row["clean_question"],
            "answer": row["clean_answer"],
            "score": float(score)
        })

    return documents

# ------------------------------------------------
# 3. RERANKING
# ------------------------------------------------

def rerank_documents(query, documents):

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
# 4. CONTEXT PREPARATION
# ------------------------------------------------

def prepare_context(documents):

    context = ""

    for i, doc in enumerate(documents, 1):

        context += (
            f"\nSOURCE {i}\n"
            f"Question: {doc['question']}\n"
            f"Answer: {doc['answer']}\n"
            f"{'-' * 60}\n"
        )

    return context

# ------------------------------------------------
# 5. PROMPT CONSTRUCTION
# ------------------------------------------------

def build_prompt(query, context):

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
3. If the context is insufficient, clearly state that.
4. Provide a clear and relevant answer.

FINAL ANSWER:
"""

    return prompt.strip()

# ------------------------------------------------
# 6. COMPLETE RAG PIPELINE
# ------------------------------------------------

if __name__ == "__main__":

    query = input("\nEnter your automotive question: ").strip()

    if not query:
        print("Please enter a question.")
        raise SystemExit

    print("\nRetrieving relevant documents...")

    documents = retrieve_documents(query)

    print("Reranking documents...")

    documents = rerank_documents(query, documents)

    print("Preparing context...")

    context = prepare_context(documents)

    final_prompt = build_prompt(query, context)

    # Save final prompt
    output_file = (
        BASE_DIR / "dataset" / "processed" /
        "automotive_final_prompt.txt"
    )

    output_file.write_text(
        final_prompt,
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("COMPLETE RAG PIPELINE OUTPUT")
    print("=" * 65)

    print(final_prompt)

    print("\nPipeline executed successfully!")
    print("Saved:", output_file)