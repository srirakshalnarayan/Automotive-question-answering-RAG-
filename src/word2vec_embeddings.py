
# ============================================================
# WORD2VEC EMBEDDINGS FOR AUTOMOTIVE NLP DATASET
# ============================================================

import os
import numpy as np
import pandas as pd
import nltk

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# ============================================================
# STEP 1: DOWNLOAD REQUIRED NLTK RESOURCES
# ============================================================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


# ============================================================
# STEP 2: WORD2VEC DEMONSTRATION
# ============================================================

print("\n========== WORD2VEC DEMONSTRATION ==========")

sample_sentences = [
    "The car engine produces power to move the vehicle.",
    "The engine requires fuel and air for combustion.",
    "Electric vehicles use batteries instead of petrol engines.",
    "The mechanic repairs the automobile engine.",
    "Vehicle maintenance improves engine performance.",
    "Cars use different types of transmission systems.",
    "The braking system helps the vehicle stop safely.",
    "Automotive manufacturers develop electric cars."
]

# Convert sentences into lowercase tokens
sample_tokens = [
    word_tokenize(sentence.lower())
    for sentence in sample_sentences
]


# ============================================================
# STEP 3: TRAIN CBOW MODEL
# ============================================================

print("\n========== CBOW MODEL ==========")

cbow_model = Word2Vec(
    sentences=sample_tokens,
    vector_size=100,
    window=3,
    min_count=1,
    sg=0,
    workers=1,
    seed=42,
    epochs=100
)

print("CBOW model trained successfully.")
print("Vocabulary size:", len(cbow_model.wv))

if "engine" in cbow_model.wv:
    print("\nWords similar to 'engine':")
    print(cbow_model.wv.most_similar("engine", topn=5))


# ============================================================
# STEP 4: TRAIN SKIP-GRAM MODEL
# ============================================================

print("\n========== SKIP-GRAM MODEL ==========")

skipgram_model = Word2Vec(
    sentences=sample_tokens,
    vector_size=100,
    window=3,
    min_count=1,
    sg=1,
    workers=1,
    seed=42,
    epochs=100
)

print("Skip-gram model trained successfully.")
print("Vocabulary size:", len(skipgram_model.wv))

if "vehicle" in skipgram_model.wv:
    print("\nWords similar to 'vehicle':")
    print(skipgram_model.wv.most_similar("vehicle", topn=5))


# ============================================================
# STEP 5: LOAD AUTOMOTIVE DATASET
# ============================================================

print("\n========== AUTOMOTIVE DATASET ==========")

input_path = "dataset/processed/automotive_cleaned_qa.csv"

if not os.path.exists(input_path):
    raise FileNotFoundError(
        f"Dataset not found: {input_path}"
    )

df = pd.read_csv(input_path)

print("Dataset shape:", df.shape)

# Select a subset to keep training manageable
df_sample = df.head(5000).copy()

print("Records selected:", len(df_sample))


# ============================================================
# STEP 6: TEXT TOKENIZATION AND PREPARATION
# ============================================================

def prepare_tokens(text):
    """
    Convert text into lowercase tokens.
    Remove punctuation, numbers and English stopwords.
    """

    if pd.isna(text):
        return []

    tokens = word_tokenize(str(text).lower())

    tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return tokens


sentences = []

for _, row in df_sample.iterrows():

    question_tokens = prepare_tokens(row["clean_question"])
    answer_tokens = prepare_tokens(row["clean_answer"])

    if question_tokens:
        sentences.append(question_tokens)

    if answer_tokens:
        sentences.append(answer_tokens)

print("Total tokenized sentences:", len(sentences))


# ============================================================
# STEP 7: TRAIN WORD2VEC ON AUTOMOTIVE DATASET
# ============================================================

print("\nTraining Word2Vec on automotive dataset...")

automotive_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1,
    workers=1,
    seed=42,
    epochs=10
)

print("Automotive Word2Vec model trained successfully.")

print("Vocabulary size:", len(automotive_model.wv))


# ============================================================
# STEP 8: FIND SIMILAR AUTOMOTIVE WORDS
# ============================================================

print("\n========== WORD SIMILARITY ==========")

test_words = [
    "engine",
    "car",
    "vehicle",
    "fuel",
    "battery"
]

for word in test_words:

    if word in automotive_model.wv:

        print(f"\nWords similar to '{word}':")

        similar_words = automotive_model.wv.most_similar(
            word,
            topn=5
        )

        for similar_word, similarity in similar_words:
            print(f"{similar_word}: {similarity:.4f}")

    else:
        print(f"\n'{word}' is not present in the vocabulary.")


# ============================================================
# STEP 9: SAVE TRAINED WORD2VEC MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

model_path = "models/automotive_word2vec.model"

automotive_model.save(model_path)

print("\nModel saved successfully:", model_path)


# ============================================================
# STEP 10: EXPORT WORD VECTORS
# ============================================================

word_vectors = []

for word in automotive_model.wv.index_to_key[:100]:

    vector = automotive_model.wv[word]

    word_vectors.append({
        "word": word,
        "vector": vector.tolist()
    })

vectors_df = pd.DataFrame(word_vectors)

os.makedirs("dataset/processed", exist_ok=True)

output_path = "dataset/processed/automotive_word_vectors.csv"

vectors_df.to_csv(output_path, index=False)

print("Word vectors saved successfully:", output_path)


# ============================================================
# STEP 11: FINAL SUMMARY
# ============================================================

print("\n========== WORD2VEC SUMMARY ==========")

print("1. CBOW implementation completed.")
print("2. Skip-gram implementation completed.")
print("3. Automotive Word2Vec model trained.")
print("4. Similarity analysis completed.")
print("5. Word embeddings exported.")

print("\nWord2Vec implementation completed successfully!")
