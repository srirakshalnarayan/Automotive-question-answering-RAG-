
# Import required libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")


# ------------------------------------------------
# PART 1: LOAD THE AUTOMOTIVE DATASET
# ------------------------------------------------

input_path = "dataset/processed/automotive_cleaned_qa.csv"

df = pd.read_csv(input_path, encoding="utf-8-sig")

print("Dataset shape:", df.shape)


# ------------------------------------------------
# PART 2: LOAD ENGLISH STOP WORDS
# ------------------------------------------------

stop_words = set(stopwords.words("english"))

print("\nTotal NLTK English stop words:", len(stop_words))

print("\nSample stop words:")
print(list(stop_words)[:30])


# ------------------------------------------------
# PART 3: SAMPLE TEXT ANALYSIS
# ------------------------------------------------

sample_text = (
    "The car engine is not starting because the battery "
    "is weak and the fuel system is not working properly."
)

tokens = word_tokenize(sample_text.lower())

# Remove stop words
filtered_tokens = [
    word for word in tokens
    if word.isalnum() and word not in stop_words
]

print("\n========== SAMPLE TEXT ==========")

print("Original text:")
print(sample_text)

print("\nOriginal tokens:")
print(tokens)

print("\nTokens after stop word removal:")
print(filtered_tokens)


# ------------------------------------------------
# PART 4: PRESERVE IMPORTANT NEGATION WORDS
# ------------------------------------------------

# Negation words are important for understanding meaning
important_words = {"not", "no", "nor", "never"}

custom_stop_words = stop_words - important_words

custom_filtered_tokens = [
    word for word in tokens
    if word.isalnum() and word not in custom_stop_words
]

print("\nTokens after custom stop word removal:")
print(custom_filtered_tokens)


# ------------------------------------------------
# PART 5: APPLY STOP WORD REMOVAL TO DATASET
# ------------------------------------------------

def remove_stopwords(text):

    tokens = word_tokenize(str(text).lower())

    filtered_tokens = [
        word for word in tokens
        if word.isalnum() and word not in custom_stop_words
    ]

    return filtered_tokens


df["question_tokens_no_stopwords"] = df["clean_question"].apply(
    remove_stopwords
)

df["answer_tokens_no_stopwords"] = df["clean_answer"].apply(
    remove_stopwords
)


# ------------------------------------------------
# PART 6: COMPARE ORIGINAL AND FILTERED TOKENS
# ------------------------------------------------

print("\n========== DATASET COMPARISON ==========")

for index, row in df.head(5).iterrows():

    print(f"\nRecord {index + 1}")

    print("\nOriginal Question:")
    print(row["clean_question"])

    print("\nTokens after stop word removal:")
    print(row["question_tokens_no_stopwords"])

    print("\nOriginal Answer:")
    print(row["clean_answer"])

    print("\nAnswer tokens after stop word removal:")
    print(row["answer_tokens_no_stopwords"])


# ------------------------------------------------
# PART 7: SAVE THE RESULTS
# ------------------------------------------------

output_path = "dataset/processed/automotive_stopwords_removed.csv"

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nStop word removal completed successfully!")
print("Output saved to:", output_path)
