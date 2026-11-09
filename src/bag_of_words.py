
# Import required libraries
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


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
# PART 2: CREATE BAG OF WORDS MODEL
# ------------------------------------------------

vectorizer = CountVectorizer()

bow_matrix = vectorizer.fit_transform(documents)

# Extract vocabulary
feature_names = vectorizer.get_feature_names_out()

print("\n========== VOCABULARY ==========")
print(feature_names)

print("\nVocabulary size:", len(feature_names))


# ------------------------------------------------
# PART 3: DISPLAY BAG OF WORDS MATRIX
# ------------------------------------------------

bow_array = bow_matrix.toarray()

bow_df = pd.DataFrame(
    bow_array,
    columns=feature_names,
    index=[
        f"Document {i}"
        for i in range(1, len(documents) + 1)
    ]
)

print("\n========== BAG OF WORDS MATRIX ==========")
print(bow_df)


# ------------------------------------------------
# PART 4: WORD FREQUENCY ANALYSIS
# ------------------------------------------------

word_frequencies = bow_df.sum(axis=0)

word_frequencies = word_frequencies.sort_values(
    ascending=False
)

print("\n========== WORD FREQUENCIES ==========")
print(word_frequencies)


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
# PART 6: SELECT SAMPLE QUESTIONS
# ------------------------------------------------

# Use a sample to keep the demonstration efficient
sample_df = df.head(5000).copy()

questions = (
    sample_df["clean_question"]
    .fillna("")
    .astype(str)
    .tolist()
)

print("Questions selected:", len(questions))


# ------------------------------------------------
# PART 7: APPLY BAG OF WORDS TO DATASET
# ------------------------------------------------

dataset_vectorizer = CountVectorizer(
    max_features=2000,
    stop_words="english"
)

dataset_bow_matrix = dataset_vectorizer.fit_transform(
    questions
)

dataset_features = dataset_vectorizer.get_feature_names_out()

print("\n========== DATASET BAG OF WORDS ==========")

print("Number of documents:", dataset_bow_matrix.shape[0])
print("Vocabulary size:", dataset_bow_matrix.shape[1])

print("\nMatrix shape:", dataset_bow_matrix.shape)


# ------------------------------------------------
# PART 8: DISPLAY SAMPLE VECTORS
# ------------------------------------------------

sample_bow = dataset_bow_matrix[:5].toarray()

sample_bow_df = pd.DataFrame(
    sample_bow,
    columns=dataset_features
)

print("\n========== FIRST FIVE DOCUMENT VECTORS ==========")
print(sample_bow_df)


# ------------------------------------------------
# PART 9: MOST FREQUENT WORDS IN DATASET
# ------------------------------------------------

dataset_word_frequency = (
    dataset_bow_matrix.sum(axis=0)
    .A1
)

frequency_df = pd.DataFrame({
    "word": dataset_features,
    "frequency": dataset_word_frequency
})

frequency_df = frequency_df.sort_values(
    by="frequency",
    ascending=False
)

print("\n========== TOP 20 FREQUENT WORDS ==========")
print(frequency_df.head(20).to_string(index=False))


# ------------------------------------------------
# PART 10: SAVE RESULTS
# ------------------------------------------------

sample_bow_df.to_csv(
    "dataset/processed/automotive_bow_sample.csv",
    index=False,
    encoding="utf-8-sig"
)

frequency_df.to_csv(
    "dataset/processed/automotive_bow_word_frequency.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nBag of Words completed successfully!")

print("Saved:")
print("1. automotive_bow_sample.csv")
print("2. automotive_bow_word_frequency.csv")
