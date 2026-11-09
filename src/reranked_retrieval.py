
# ============================================================
# CROSS-ENCODER RERANKING
# AUTOMOTIVE NLP AND RAG PROJECT
# ============================================================

import os
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

INDEX_PATH = "vector_store/automotive_faiss.index"
METADATA_PATH = "vector_store/automotive_metadata.csv"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_TOP_K = 20
FINAL_TOP_K = 3


# ============================================================
# STEP 2: LOAD FAISS INDEX AND METADATA
# ============================================================

print("\n========== LOADING AUTOMOTIVE KNOWLEDGE BASE ==========")

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError("FAISS index not found.")

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("Metadata file not found.")

index = faiss.read_index(INDEX_PATH)

df = pd.read_csv(METADATA_PATH)

if len(df) != index.ntotal:
    raise ValueError("Index and metadata counts do not match.")

print("FAISS index loaded.")
print("Available records:", len(df))


# ============================================================
# STEP 3: LOAD MODELS
# ============================================================

print("\n========== LOADING NLP MODELS ==========")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Sentence embedding model loaded.")

reranker = CrossEncoder(RERANKER_MODEL)

print("Cross-Encoder reranking model loaded.")


# ============================================================
# STEP 4: INITIAL FAISS RETRIEVAL
# ============================================================

def retrieve_candidates(query, top_k=INITIAL_TOP_K):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    candidates = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        candidates.append({
            "question": str(df.iloc[idx]["clean_question"]),
            "answer": str(df.iloc[idx]["clean_answer"]),
            "faiss_score": float(score),
            "metadata_index": int(idx)
        })

    return candidates


# ============================================================
# STEP 5: CROSS-ENCODER RERANKING
# ============================================================

def rerank_candidates(query, candidates):

    if not candidates:
        return []

    # Compare the query against each retrieved question
    pairs = [
        [query, candidate["question"]]
        for candidate in candidates
    ]

    # Cross-Encoder calculates relevance scores
    rerank_scores = reranker.predict(
        pairs,
        show_progress_bar=False
    )

    for candidate, score in zip(candidates, rerank_scores):

        candidate["rerank_score"] = float(score)

    # Sort according to Cross-Encoder scores
    reranked_results = sorted(
        candidates,
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return reranked_results[:FINAL_TOP_K]


# ============================================================
# STEP 6: COMPLETE RETRIEVAL PIPELINE
# ============================================================

def search_automotive_knowledge(query):

    if not query.strip():
        return []

    # First-stage retrieval
    candidates = retrieve_candidates(
        query,
        top_k=INITIAL_TOP_K
    )

    # Second-stage reranking
    final_results = rerank_candidates(
        query,
        candidates
    )

    return final_results


# ============================================================
# STEP 7: DISPLAY RESULTS
# ============================================================

def display_results(query, results):

    print("\n" + "=" * 70)
    print("AUTOMOTIVE QUESTION SEARCH")
    print("=" * 70)

    print("\nUser Query:", query)

    if not results:
        print("\nNo results available.")
        return

    print("\nTop relevant results:\n")

    for rank, result in enumerate(results, start=1):

        print(f"[{rank}] {result['question']}")

        answer = result["answer"]

        # Display a short answer preview
        preview_length = 200

        if len(answer) > preview_length:
            answer = answer[:preview_length].rsplit(" ", 1)[0] + "..."

        print("    Answer:", answer)

        print(
            "    FAISS Similarity:",
            round(result["faiss_score"], 4)
        )

        print(
            "    Reranker Score:",
            round(result["rerank_score"], 4)
        )

        print("-" * 70)


# ============================================================
# STEP 8: INTERACTIVE AUTOMOTIVE SEARCH
# ============================================================

if __name__ == "__main__":

    print("\n========== RERANKED AUTOMOTIVE RETRIEVAL ==========")

    print("Ask an automotive question.")
    print("Type 'exit' to close.")

    while True:

        user_query = input("\nYour question: ").strip()

        if user_query.lower() == "exit":
            print("\nRetrieval system closed.")
            break

        if not user_query:
            print("Please enter a valid question.")
            continue

        results = search_automotive_knowledge(user_query)

        display_results(user_query, results)


# ============================================================
# END OF PROGRAM
# ============================================================
