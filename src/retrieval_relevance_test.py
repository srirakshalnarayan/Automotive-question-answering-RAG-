
import pandas as pd
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer

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
    "retrieval_relevance_results.csv"
)

# ------------------------------------------------
# 2. LOAD DATA AND MODELS
# ------------------------------------------------

df = pd.read_csv(METADATA_FILE)

index = faiss.read_index(str(INDEX_FILE))

model = SentenceTransformer("all-MiniLM-L6-v2")

# Select five actual questions from the dataset
test_records = df.head(5).copy()

# ------------------------------------------------
# 3. RETRIEVAL FUNCTION
# ------------------------------------------------

def retrieve(query, top_k=5):

    query_embedding = model.encode(
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

    return indices[0], scores[0]

# ------------------------------------------------
# 4. EVALUATION
# ------------------------------------------------

results = []

hits = 0
reciprocal_ranks = []

print("\nRETRIEVAL RELEVANCE EVALUATION")
print("=" * 75)

for test_position, row in test_records.iterrows():

    query = row["clean_question"]

    # embedding_index is the ground-truth FAISS position
    correct_index = int(row["embedding_index"])

    retrieved_indices, scores = retrieve(query)

    # Check whether the correct record was retrieved
    matching_positions = np.where(
        retrieved_indices == correct_index
    )[0]

    if len(matching_positions) > 0:

        rank = int(matching_positions[0]) + 1

        hit = 1
        reciprocal_rank = 1 / rank

        hits += 1

    else:

        rank = None
        hit = 0
        reciprocal_rank = 0

    reciprocal_ranks.append(reciprocal_rank)

    results.append({
        "question": query,
        "correct_index": correct_index,
        "retrieved_indices": retrieved_indices.tolist(),
        "rank_of_correct_record": rank,
        "hit_at_5": hit,
        "reciprocal_rank": reciprocal_rank
    })

    print(f"\nQuestion {test_position + 1}: {query}")
    print("Correct FAISS index:", correct_index)
    print("Retrieved indices:", retrieved_indices.tolist())
    print("Rank of correct record:", rank)
    print("Hit@5:", hit)

# ------------------------------------------------
# 5. CALCULATE METRICS
# ------------------------------------------------

total_questions = len(test_records)

hit_rate = hits / total_questions

mrr = np.mean(reciprocal_ranks)

print("\n" + "=" * 75)
print("FINAL EVALUATION RESULTS")
print("=" * 75)

print(f"Total test questions: {total_questions}")
print(f"Successful retrievals: {hits}")
print(f"Hit Rate@5: {hit_rate:.4f}")
print(f"MRR: {mrr:.4f}")

# ------------------------------------------------
# 6. SAVE RESULTS
# ------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nEvaluation completed!")
print("Results saved to:", OUTPUT_FILE)
