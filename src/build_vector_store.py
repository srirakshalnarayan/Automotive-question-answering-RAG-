
# ============================================================
# BUILD FAISS VECTOR STORE
# AUTOMOTIVE NLP AND RAG PROJECT
# ============================================================

import os
import numpy as np
import pandas as pd
import faiss


# ============================================================
# STEP 1: LOAD SENTENCE EMBEDDINGS
# ============================================================

print("\n========== LOADING SENTENCE EMBEDDINGS ==========")

embedding_path = "models/automotive_question_embeddings.npy"

if not os.path.exists(embedding_path):
    raise FileNotFoundError(
        "Sentence embeddings not found. "
        "Run sentence_embeddings.py first."
    )

embeddings = np.load(embedding_path)

print("Embedding shape:", embeddings.shape)
print("Number of embeddings:", len(embeddings))


# ============================================================
# STEP 2: LOAD QUESTION-ANSWER MAPPING
# ============================================================

print("\n========== LOADING QUESTION-ANSWER DATA ==========")

mapping_path = (
    "dataset/processed/automotive_embedding_mapping.csv"
)

if not os.path.exists(mapping_path):
    raise FileNotFoundError(
        "Embedding mapping file not found."
    )

df = pd.read_csv(mapping_path)

print("Mapping shape:", df.shape)

if len(df) != len(embeddings):
    raise ValueError(
        "Embedding count and mapping count do not match."
    )

if not np.array_equal(
    df["embedding_index"].to_numpy(),
    np.arange(len(df))
):
    raise ValueError(
        "Embedding indices are not in the expected order."
    )


# ============================================================
# STEP 3: PREPARE EMBEDDINGS
# ============================================================

print("\n========== PREPARING VECTORS ==========")

# FAISS expects float32 vectors
embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)

# Normalize vectors for cosine similarity
faiss.normalize_L2(embeddings)

vector_dimension = embeddings.shape[1]

print("Vector dimension:", vector_dimension)
print("Vectors normalized successfully.")


# ============================================================
# STEP 4: CREATE FAISS INDEX
# ============================================================

print("\n========== CREATING FAISS INDEX ==========")

# IndexFlatIP uses inner product.
# With normalized vectors, this equals cosine similarity.

index = faiss.IndexFlatIP(vector_dimension)

print("FAISS index created successfully.")


# ============================================================
# STEP 5: ADD EMBEDDINGS TO THE INDEX
# ============================================================

print("\n========== ADDING VECTORS ==========")

index.add(embeddings)

print("Vectors added successfully.")
print("Total vectors in index:", index.ntotal)


# ============================================================
# STEP 6: SAVE FAISS INDEX
# ============================================================

print("\n========== SAVING VECTOR STORE ==========")

os.makedirs("vector_store", exist_ok=True)

index_path = "vector_store/automotive_faiss.index"

faiss.write_index(index, index_path)

print("FAISS index saved at:", index_path)


# ============================================================
# STEP 7: SAVE ASSOCIATED METADATA
# ============================================================

metadata_path = "vector_store/automotive_metadata.csv"

df.to_csv(metadata_path, index=False)

print("Metadata saved at:", metadata_path)


# ============================================================
# STEP 8: TEST VECTOR SEARCH
# ============================================================

print("\n========== TESTING VECTOR SEARCH ==========")

# Use the first stored embedding as a test query
test_vector = embeddings[0].reshape(1, -1)

scores, indices = index.search(test_vector, 5)

print("\nTest Question:")
print(df.iloc[0]["clean_question"])

print("\nTop 5 Retrieved Results:\n")

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    if idx == -1:
        continue

    print(f"Rank: {rank}")
    print("Question:", df.iloc[idx]["clean_question"])
    print("Answer:", df.iloc[idx]["clean_answer"])
    print("Cosine Similarity:", round(float(score), 4))
    print("-" * 70)


# ============================================================
# STEP 9: FINAL SUMMARY
# ============================================================

print("\n========== VECTOR STORE SUMMARY ==========")

print("Vector database: FAISS")
print("Index type: IndexFlatIP")
print("Similarity: Cosine Similarity")
print("Total stored vectors:", index.ntotal)
print("Vector dimension:", vector_dimension)

print("\nFAISS vector store created successfully!")
