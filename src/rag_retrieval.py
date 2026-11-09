
# ============================================================
# AUTOMOTIVE RAG - RETRIEVAL COMPONENT
# ============================================================

import os
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer


# ============================================================
# STEP 1: LOAD FAISS VECTOR STORE
# ============================================================

INDEX_PATH = "vector_store/automotive_faiss.index"
METADATA_PATH = "vector_store/automotive_metadata.csv"

print("\nLoading automotive knowledge base...")

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError("FAISS index not found.")

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("Metadata file not found.")

index = faiss.read_index(INDEX_PATH)
df = pd.read_csv(METADATA_PATH)

if len(df) != index.ntotal:
    raise ValueError("FAISS index and metadata do not match.")

print("Knowledge base loaded successfully.")
print("Available records:", len(df))


# ============================================================
# STEP 2: LOAD SENTENCE TRANSFORMER
# ============================================================

print("\nLoading sentence transformer...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully.")


# ============================================================
# STEP 3: RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(query, top_k=3):

    if not query.strip():
        return []

    query_embedding = model.encode(
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

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "question": df.iloc[idx]["clean_question"],
            "answer": df.iloc[idx]["clean_answer"],
            "similarity": float(score)
        })

    return results


# ============================================================
# STEP 4: DISPLAY CLEAN SEARCH RESULTS
# ============================================================

def display_results(query, results):

    print("\n" + "=" * 65)
    print("AUTOMOTIVE KNOWLEDGE SEARCH")
    print("=" * 65)

    print("\nYour question:", query)

    if not results:
        print("\nNo matching records found.")
        return

    print(f"\nTop {len(results)} relevant results:\n")

    for rank, result in enumerate(results, start=1):

        question = str(result["question"])
        answer = str(result["answer"])

        # Short answer preview
        preview_length = 220

        if len(answer) > preview_length:
            answer = answer[:preview_length].rsplit(" ", 1)[0] + "..."

        print(f"[{rank}] {question}")

        print("    Answer:", answer)

        print(
            "    Similarity:",
            f"{result['similarity']:.4f}"
        )

        print("-" * 65)


# ============================================================
# STEP 5: INTERACTIVE SEARCH
# ============================================================

if __name__ == "__main__":

    print("\n========== AUTOMOTIVE QUESTION RETRIEVAL ==========")

    print("Ask any automotive-related question.")
    print("Type 'exit' to close the application.")

    while True:

        user_query = input("\nYour question: ").strip()

        if user_query.lower() == "exit":
            print("\nAutomotive retrieval system closed.")
            break

        if not user_query:
            print("Please enter a question.")
            continue

        results = retrieve_documents(
            user_query,
            top_k=3
        )

        display_results(user_query, results)
