
import pandas as pd
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

# ------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    BASE_DIR / "vector_store" / "automotive_metadata.csv"
)

INDEX_FILE = (
    BASE_DIR / "vector_store" / "automotive_faiss.index"
)

OUTPUT_FILE = (
    BASE_DIR / "dataset" / "processed" /
    "last_rag_context.txt"
)

# ------------------------------------------------
# 2. LOAD DATA AND MODELS
# ------------------------------------------------

print("Loading automotive metadata...")

df = pd.read_csv(METADATA_FILE)

print("Loading FAISS index...")

index = faiss.read_index(str(INDEX_FILE))

print("Loading sentence embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Loading cross-encoder reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("\nAll components loaded successfully!")

# ------------------------------------------------
# 3. RETRIEVE CANDIDATES USING FAISS
# ------------------------------------------------

def retrieve_candidates(query, top_k=20):

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
        top_k
    )

    candidates = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        row = df.iloc[idx]

        candidates.append({
            "question": row["clean_question"],
            "answer": row["clean_answer"],
            "faiss_score": float(score)
        })

    return candidates

# ------------------------------------------------
# 4. RERANK RETRIEVED RESULTS
# ------------------------------------------------

def rerank_candidates(query, candidates, top_k=3):

    pairs = [
        [query, item["question"]]
        for item in candidates
    ]

    scores = reranker.predict(pairs)

    for item, score in zip(candidates, scores):
        item["reranker_score"] = float(score)

    ranked_results = sorted(
        candidates,
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return ranked_results[:top_k]

# ------------------------------------------------
# 5. PREPARE STRUCTURED CONTEXT
# ------------------------------------------------

def prepare_context(results):

    context = ""

    for i, item in enumerate(results, start=1):

        context += (
            f"\nSOURCE {i}\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}\n"
            f"Relevance score: "
            f"{item['reranker_score']:.4f}\n"
            f"{'-' * 70}\n"
        )

    return context.strip()

# ------------------------------------------------
# 6. MAIN PROGRAM
# ------------------------------------------------

if __name__ == "__main__":

    query = input(
        "\nEnter your automotive question: "
    ).strip()

    if not query:
        raise ValueError("Please enter a question.")

    print("\nRetrieving relevant records...")

    candidates = retrieve_candidates(query)

    if not candidates:
        print("No relevant records found.")
        raise SystemExit

    print("Reranking retrieved records...")

    results = rerank_candidates(
        query,
        candidates,
        top_k=3
    )

    print("\nPreparing RAG context...")

    context = prepare_context(results)

    # Save context
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        context,
        encoding="utf-8"
    )

    # Display results
    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)

    print(context)

    print("\nContext saved successfully!")
    print("Output:", OUTPUT_FILE)
