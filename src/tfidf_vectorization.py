
# Import required libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------------------------------
# PART 1: SAMPLE AUTOMOTIVE DOCUMENTS
# ------------------------------------------------

documents = [
    "The car engine needs regular maintenance",
    "The car engine requires repair",
    "Regular maintenance improves engine performance",
    "Electric vehicles use rechargeable batteries",
    "The vehicle battery requires maintenance"
]

print("\n========== SAMPLE DOCUMENTS ==========")

for i, document in enumerate(documents, start=1):
    print(f"Document {i}: {document}")


# ------------------------------------------------
# PART 2: CREATE TF-IDF VECTORIZER
# ------------------------------------------------

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

print("\n========== VOCABULARY ==========")
print(feature_names)

print("\nVocabulary size:", len(feature_names))


# ------------------------------------------------
# PART 3: DISPLAY TF-IDF MATRIX
# ------------------------------------------------

tfidf_array = tfidf_matrix.toarray()

tfidf_df = pd.DataFrame(
    tfidf_array,
    columns=feature_names,
    index=[
        f"Document {i}"
        for i in range(1, len(documents) + 1)
    ]
)

print("\n========== TF-IDF MATRIX ==========")
print(tfidf_df.round(3))


# ------------------------------------------------
# PART 4: DISPLAY IMPORTANT WORDS
# ------------------------------------------------

print("\n========== IMPORTANT WORDS IN EACH DOCUMENT ==========")

for index, row in tfidf_df.iterrows():

    important_words = row.sort_values(
        ascending=False
    ).head(3)

    print(f"\n{index}:")

    for word, score in important_words.items():
        print(f"{word}: {score:.3f}")


# ------------------------------------------------
# PART 5: LOAD AUTOMOTIVE DATASET
# ------------------------------------------------

input_path = "dataset/processed/automotive_cleaned_qa.csv"

df = pd.read_csv(
    input_path,
    encoding="utf-8-sig"
)

print("\n========== DATASET INFORMATION ==========")
print("Original dataset shape:", df.shape)


# ------------------------------------------------
# PART 6: SELECT DATASET SAMPLE
# ------------------------------------------------

# Process 5000 questions for this experiment
sample_df = df.head(5000).copy()

questions = (
    sample_df["clean_question"]
    .fillna("")
    .astype(str)
    .tolist()
)

print("Questions selected:", len(questions))


# ------------------------------------------------
# PART 7: APPLY TF-IDF TO AUTOMOTIVE QUESTIONS
# ------------------------------------------------

dataset_vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words="english"
)

dataset_tfidf_matrix = dataset_vectorizer.fit_transform(
    questions
)

dataset_features = dataset_vectorizer.get_feature_names_out()

print("\n========== DATASET TF-IDF ==========")

print("Number of documents:", dataset_tfidf_matrix.shape[0])
print("Vocabulary size:", dataset_tfidf_matrix.shape[1])
print("Matrix shape:", dataset_tfidf_matrix.shape)


# ------------------------------------------------
# PART 8: DISPLAY SAMPLE TF-IDF VECTORS
# ------------------------------------------------

sample_tfidf = dataset_tfidf_matrix[:5].toarray()

sample_tfidf_df = pd.DataFrame(
    sample_tfidf,
    columns=dataset_features
)

print("\n========== FIRST FIVE TF-IDF VECTORS ==========")
print(sample_tfidf_df.round(3))


# ------------------------------------------------
# PART 9: IDENTIFY TOP TF-IDF TERMS
# ------------------------------------------------

average_scores = dataset_tfidf_matrix.mean(axis=0).A1

importance_df = pd.DataFrame({
    "word": dataset_features,
    "average_tfidf": average_scores
})

importance_df = importance_df.sort_values(
    by="average_tfidf",
    ascending=False
)

print("\n========== TOP 20 TF-IDF TERMS ==========")
print(importance_df.head(20).to_string(index=False))


# ------------------------------------------------
# PART 10: SAVE RESULTS
# ------------------------------------------------

sample_tfidf_df.to_csv(
    "dataset/processed/automotive_tfidf_sample.csv",
    index=False,
    encoding="utf-8-sig"
)

importance_df.to_csv(
    "dataset/processed/automotive_tfidf_importance.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTF-IDF vectorization completed successfully!")

print("Saved:")
print("1. automotive_tfidf_sample.csv")
print("2. automotive_tfidf_importance.csv")
