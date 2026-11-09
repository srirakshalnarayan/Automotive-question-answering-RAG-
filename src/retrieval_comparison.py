
# ============================================================
# TF-IDF VS SEMANTIC RETRIEVAL COMPARISON
# AUTOMOTIVE NLP PROJECT
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ============================================================
# STEP 1: LOAD AUTOMOTIVE DATASET
# ============================================================

print("\n========== LOADING AUTOMOTIVE DATASET ==========")

dataset_path = "dataset/processed/automotive_cleaned_qa.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

df = pd.read_csv(dataset_path)

# Use the same first 5000 records as the embedding experiment
df = df.head(5000).copy()
df = df.reset_index(drop=True)

df["clean_question"] = df["clean_question"].fillna("").astype(str)
df["clean_answer"] = df["clean_answer"].fillna("").astype(str)

print("Dataset shape:", df.shape)


# ============================================================
# STEP 2: LOAD SEMANTIC EMBEDDINGS
# ============================================================

print("\n========== LOADING SENTENCE EMBEDDINGS ==========")

embedding_path = "models/automotive_question_embeddings.npy"

if not os.path.exists(embedding_path):
    raise FileNotFoundError(
        "Sentence embeddings not found. "
        "Run sentence_embeddings.py first."
    )

semantic_embeddings = np.load(embedding_path)

if len(semantic_embeddings) != len(df):
    raise ValueError(
        "Embedding count does not match the dataset. "
        "Regenerate sentence embeddings using the same dataset records."
    )

print("Semantic embedding shape:", semantic_embeddings.shape)


# ============================================================
# STEP 3: BUILD TF-IDF REPRESENTATION
# ============================================================

print("\n========== BUILDING TF-IDF MODEL ==========")

questions = df["clean_question"].tolist()

tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf_vectorizer.fit_transform(questions)

print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("TF-IDF vocabulary size:", len(tfidf_vectorizer.vocabulary_))


# ============================================================
# STEP 4: LOAD SENTENCE TRANSFORMER
# ============================================================

print("\n========== LOADING SEMANTIC MODEL ==========")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Sentence Transformer loaded successfully.")


# ============================================================
# STEP 5: DEFINE RETRIEVAL FUNCTIONS
# ============================================================

def tfidf_retrieval(query, top_k=5):

    query_vector = tfidf_vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "question": questions[index],
            "answer": df.iloc[index]["clean_answer"],
            "similarity": float(scores[index])
        })

    return results


def semantic_retrieval(query, top_k=5):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    scores = cosine_similarity(
        query_embedding,
        semantic_embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "question": questions[index],
            "answer": df.iloc[index]["clean_answer"],
            "similarity": float(scores[index])
        })

    return results


# ============================================================
# STEP 6: TEST RETRIEVAL USING AUTOMOTIVE QUERIES
# ============================================================

test_queries = [
    "Why is my car engine overheating?",
    "How can I improve fuel efficiency?",
    "What causes a vehicle battery to drain?",
    "Why are my car brakes making noise?",
    "How does an electric vehicle battery work?"
]

comparison_results = []

for query_number, query in enumerate(test_queries, start=1):

    print("\n" + "=" * 80)
    print(f"QUERY {query_number}: {query}")
    print("=" * 80)

    # TF-IDF results
    tfidf_results = tfidf_retrieval(query, top_k=5)

    print("\n--- TF-IDF RETRIEVAL ---")

    for rank, result in enumerate(tfidf_results, start=1):

        print(f"\nRank {rank}")
        print("Question:", result["question"])
        print("Similarity:", round(result["similarity"], 4))
        print("Answer:", result["answer"][:300])

        comparison_results.append({
            "query": query,
            "method": "TF-IDF",
            "rank": rank,
            "retrieved_question": result["question"],
            "similarity_score": result["similarity"],
            "retrieved_answer": result["answer"]
        })

    # Semantic results
    semantic_results = semantic_retrieval(query, top_k=5)

    print("\n--- SEMANTIC RETRIEVAL ---")

    for rank, result in enumerate(semantic_results, start=1):

        print(f"\nRank {rank}")
        print("Question:", result["question"])
        print("Similarity:", round(result["similarity"], 4))
        print("Answer:", result["answer"][:300])

        comparison_results.append({
            "query": query,
            "method": "Semantic",
            "rank": rank,
            "retrieved_question": result["question"],
            "similarity_score": result["similarity"],
            "retrieved_answer": result["answer"]
        })


# ============================================================
# STEP 7: SAVE COMPARISON RESULTS
# ============================================================

print("\n========== SAVING RESULTS ==========")

os.makedirs("dataset/processed", exist_ok=True)

results_df = pd.DataFrame(comparison_results)

output_path = "dataset/processed/retrieval_comparison_results.csv"

results_df.to_csv(output_path, index=False)

print("Comparison results saved at:", output_path)


# ============================================================
# STEP 8: DISPLAY SUMMARY
# ============================================================

print("\n========== RETRIEVAL COMPARISON SUMMARY ==========")

summary = (
    results_df.groupby("method")["similarity_score"]
    .agg(["mean", "max", "min"])
    .round(4)
)

print(summary)

print("\nTotal queries tested:", len(test_queries))
print("Retrieval methods compared: TF-IDF and Semantic")
print("Top results per method per query: 5")

print("\nRetrieval comparison completed successfully!")
