
# ============================================================
# SENTENCE EMBEDDINGS AND SEMANTIC SIMILARITY
# AUTOMOTIVE NLP PROJECT
# ============================================================

import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# STEP 1: LOAD PRETRAINED SENTENCE TRANSFORMER MODEL
# ============================================================

print("\n========== SENTENCE EMBEDDINGS ==========")

model_name = "all-MiniLM-L6-v2"

print("Loading model:", model_name)

model = SentenceTransformer(model_name)

print("Sentence Transformer model loaded successfully.")


# ============================================================
# STEP 2: DEMONSTRATE SENTENCE EMBEDDINGS
# ============================================================

sample_sentences = [
    "My car engine is overheating.",
    "The vehicle motor is getting too hot.",
    "Electric cars use rechargeable batteries.",
    "The braking system helps stop a vehicle.",
    "A mechanic repairs damaged automobile engines."
]

print("\nGenerating sentence embeddings...")

sample_embeddings = model.encode(sample_sentences)

print("Number of sentences:", len(sample_sentences))
print("Embedding shape:", sample_embeddings.shape)

print("\nFirst sentence embedding (first 10 values):")
print(sample_embeddings[0][:10])


# ============================================================
# STEP 3: CALCULATE SEMANTIC SIMILARITY
# ============================================================

print("\n========== SEMANTIC SIMILARITY ==========")

similarity_matrix = cosine_similarity(sample_embeddings)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=sample_sentences,
    columns=sample_sentences
)

print("\nCosine Similarity Matrix:")
print(similarity_df.round(3).to_string())


# ============================================================
# STEP 4: COMPARE SENTENCE SIMILARITY
# ============================================================

sentence_1 = "My car engine is overheating."
sentence_2 = "The vehicle motor is getting too hot."

index_1 = sample_sentences.index(sentence_1)
index_2 = sample_sentences.index(sentence_2)

similarity_score = similarity_matrix[index_1][index_2]

print("\nSimilarity between:")
print("Sentence 1:", sentence_1)
print("Sentence 2:", sentence_2)

print("Cosine similarity:", round(float(similarity_score), 4))


# ============================================================
# STEP 5: LOAD AUTOMOTIVE DATASET
# ============================================================

print("\n========== AUTOMOTIVE DATASET ==========")

input_path = "dataset/processed/automotive_cleaned_qa.csv"

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Dataset not found: {input_path}")

df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# Select first 5000 records for manageable processing
df_sample = df.head(5000).copy()

print("Records selected:", len(df_sample))


# ============================================================
# STEP 6: GENERATE AUTOMOTIVE QUESTION EMBEDDINGS
# ============================================================

print("\nGenerating automotive question embeddings...")

questions = df_sample["clean_question"].fillna("").astype(str).tolist()

question_embeddings = model.encode(
    questions,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Question embedding generation completed.")
print("Embedding matrix shape:", question_embeddings.shape)


# ============================================================
# STEP 7: SAVE EMBEDDINGS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("dataset/processed", exist_ok=True)

embedding_path = "models/automotive_question_embeddings.npy"

np.save(embedding_path, question_embeddings)

print("\nEmbeddings saved at:", embedding_path)


# ============================================================
# STEP 8: SAVE QUESTION AND EMBEDDING MAPPING
# ============================================================

mapping_df = df_sample[
    ["clean_question", "clean_answer"]
].copy()

mapping_df["embedding_index"] = range(len(mapping_df))

mapping_path = "dataset/processed/automotive_embedding_mapping.csv"

mapping_df.to_csv(mapping_path, index=False)

print("Question mapping saved at:", mapping_path)


# ============================================================
# STEP 9: TEST SEMANTIC RETRIEVAL
# ============================================================

print("\n========== SEMANTIC SEARCH TEST ==========")

query = "Why does my car engine become too hot?"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
)

similarity_scores = cosine_similarity(
    query_embedding,
    question_embeddings
)[0]

top_indices = np.argsort(similarity_scores)[::-1][:5]

print("\nUser Query:", query)
print("\nTop 5 semantically similar questions:\n")

for rank, index in enumerate(top_indices, start=1):

    print(f"Rank {rank}")
    print("Question:", questions[index])
    print("Similarity:", round(float(similarity_scores[index]), 4))
    print("Answer:", df_sample.iloc[index]["clean_answer"])
    print("-" * 70)


# ============================================================
# STEP 10: FINAL SUMMARY
# ============================================================

print("\n========== FINAL SUMMARY ==========")

print("1. Pretrained Sentence Transformer loaded.")
print("2. Sentence embeddings generated.")
print("3. Cosine similarity calculated.")
print("4. Automotive question embeddings generated.")
print("5. Semantic retrieval tested.")

print("\nSentence Embeddings implementation completed!")
