
# Import required libraries
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter


# ------------------------------------------------
# PART 1: DOWNLOAD NLTK RESOURCES
# ------------------------------------------------

nltk.download("punkt")
nltk.download("punkt_tab")


# ------------------------------------------------
# PART 2: SAMPLE AUTOMOTIVE TEXT
# ------------------------------------------------

sample_text = """
The car engine requires regular maintenance.
The car engine uses fuel to produce energy.
Regular maintenance improves engine performance.
"""

print("\n========== SAMPLE TEXT ==========")
print(sample_text)


# ------------------------------------------------
# PART 3: TOKENIZATION
# ------------------------------------------------

tokens = word_tokenize(sample_text.lower())

words = [
    word for word in tokens
    if word.isalpha()
]

print("\n========== TOKENS ==========")
print(words)


# ------------------------------------------------
# PART 4: UNIGRAM ANALYSIS
# ------------------------------------------------

unigrams = list(ngrams(words, 1))

unigram_frequency = Counter(unigrams)

print("\n========== UNIGRAM FREQUENCY ==========")

for gram, count in unigram_frequency.most_common(15):
    print(" ".join(gram), ":", count)


# ------------------------------------------------
# PART 5: BIGRAM ANALYSIS
# ------------------------------------------------

bigrams = list(ngrams(words, 2))

bigram_frequency = Counter(bigrams)

print("\n========== BIGRAM FREQUENCY ==========")

for gram, count in bigram_frequency.most_common(15):
    print(" ".join(gram), ":", count)


# ------------------------------------------------
# PART 6: TRIGRAM ANALYSIS
# ------------------------------------------------

trigrams = list(ngrams(words, 3))

trigram_frequency = Counter(trigrams)

print("\n========== TRIGRAM FREQUENCY ==========")

for gram, count in trigram_frequency.most_common(15):
    print(" ".join(gram), ":", count)


# ------------------------------------------------
# PART 7: LOAD AUTOMOTIVE DATASET
# ------------------------------------------------

input_path = "dataset/processed/automotive_cleaned_qa.csv"

df = pd.read_csv(
    input_path,
    encoding="utf-8-sig"
)

print("\n========== DATASET INFORMATION ==========")
print("Dataset shape:", df.shape)


# ------------------------------------------------
# PART 8: GENERATE N-GRAMS FROM QUESTIONS
# ------------------------------------------------

def generate_ngrams(text, n):

    tokens = word_tokenize(str(text).lower())

    words = [
        word for word in tokens
        if word.isalpha()
    ]

    return list(ngrams(words, n))


# Use the first 5000 questions for efficient analysis
sample_df = df.head(5000).copy()

sample_df["unigrams"] = sample_df["clean_question"].apply(
    lambda text: generate_ngrams(text, 1)
)

sample_df["bigrams"] = sample_df["clean_question"].apply(
    lambda text: generate_ngrams(text, 2)
)

sample_df["trigrams"] = sample_df["clean_question"].apply(
    lambda text: generate_ngrams(text, 3)
)


# ------------------------------------------------
# PART 9: DATASET N-GRAM FREQUENCY ANALYSIS
# ------------------------------------------------

all_unigrams = [
    gram
    for row in sample_df["unigrams"]
    for gram in row
]

all_bigrams = [
    gram
    for row in sample_df["bigrams"]
    for gram in row
]

all_trigrams = [
    gram
    for row in sample_df["trigrams"]
    for gram in row
]


print("\n========== DATASET TOP UNIGRAMS ==========")

for gram, count in Counter(all_unigrams).most_common(20):
    print(" ".join(gram), ":", count)


print("\n========== DATASET TOP BIGRAMS ==========")

for gram, count in Counter(all_bigrams).most_common(20):
    print(" ".join(gram), ":", count)


print("\n========== DATASET TOP TRIGRAMS ==========")

for gram, count in Counter(all_trigrams).most_common(20):
    print(" ".join(gram), ":", count)


# ------------------------------------------------
# PART 10: DISPLAY SAMPLE RECORDS
# ------------------------------------------------

print("\n========== SAMPLE N-GRAM RESULTS ==========")

for index, row in sample_df.head(5).iterrows():

    print(f"\nRecord {index + 1}")

    print("Question:", row["clean_question"])
    print("Unigrams:", row["unigrams"])
    print("Bigrams:", row["bigrams"])
    print("Trigrams:", row["trigrams"])


# ------------------------------------------------
# PART 11: SAVE RESULTS
# ------------------------------------------------

output_path = "dataset/processed/automotive_ngram_analysis.csv"

sample_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nN-gram analysis completed successfully!")

print("Records processed:", len(sample_df))
print("Output saved to:", output_path)
