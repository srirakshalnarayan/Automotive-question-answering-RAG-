
# Import required libraries
import pandas as pd
import re
import unicodedata

# ------------------------------------------------
# PART 1: LOAD THE EXTRACTED DATASET
# ------------------------------------------------

input_path = "dataset/processed/automotive_english_qa.csv"

df = pd.read_csv(input_path, encoding="utf-8-sig")

print("Original dataset shape:", df.shape)

print("\nFirst 5 records:")
print(df[["question", "answer"]].head())


# ------------------------------------------------
# PART 2: CHECK MISSING AND EMPTY VALUES
# ------------------------------------------------

print("\nMissing values:")
print(df[["question", "answer"]].isnull().sum())

# Remove rows where question or answer is missing
df = df.dropna(subset=["question", "answer"]).copy()

# Remove rows where question or answer contains only whitespace
df = df[
    (df["question"].astype(str).str.strip() != "") &
    (df["answer"].astype(str).str.strip() != "")
].copy()

print("\nShape after removing missing and empty records:", df.shape)


# ------------------------------------------------
# PART 3: TEXT CLEANING FUNCTION
# ------------------------------------------------

def clean_text(text):

    # Convert text to string
    text = str(text)

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove unwanted control characters
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Cc" or char in "\n\t"
    )

    # Replace line breaks and tabs with spaces
    text = re.sub(r"[\n\t\r]+", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces at the beginning and end
    text = text.strip()

    return text


# ------------------------------------------------
# PART 4: APPLY TEXT CLEANING
# ------------------------------------------------

# Preserve original question and answer columns
df["clean_question"] = df["question"].apply(clean_text)
df["clean_answer"] = df["answer"].apply(clean_text)

print("\nOriginal and cleaned text comparison:")

for i in range(min(5, len(df))):

    print("\nRecord:", i + 1)

    print("Original Question:", df.iloc[i]["question"])
    print("Cleaned Question:", df.iloc[i]["clean_question"])

    print("Original Answer:", df.iloc[i]["answer"])
    print("Cleaned Answer:", df.iloc[i]["clean_answer"])


# ------------------------------------------------
# PART 5: DUPLICATE ANALYSIS
# ------------------------------------------------

# Count exact duplicate question-answer pairs
duplicate_pairs = df.duplicated(
    subset=["clean_question", "clean_answer"]
).sum()

print("\nDuplicate question-answer pairs:", duplicate_pairs)

# Count repeated questions, even if answers differ
duplicate_questions = df.duplicated(
    subset=["clean_question"]
).sum()

print("Repeated questions:", duplicate_questions)

# Remove only exact duplicate question-answer pairs
df = df.drop_duplicates(
    subset=["clean_question", "clean_answer"],
    keep="first"
).copy()

print("\nShape after removing exact duplicate pairs:", df.shape)


# ------------------------------------------------
# PART 6: TEXT LENGTH ANALYSIS
# ------------------------------------------------

df["question_length"] = df["clean_question"].str.len()
df["answer_length"] = df["clean_answer"].str.len()

print("\nQuestion length statistics:")
print(df["question_length"].describe())

print("\nAnswer length statistics:")
print(df["answer_length"].describe())


# ------------------------------------------------
# PART 7: SAVE THE CLEANED DATASET
# ------------------------------------------------

output_path = "dataset/processed/automotive_cleaned_qa.csv"

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nPreprocessing completed successfully!")
print("Final dataset shape:", df.shape)
print("Cleaned dataset saved to:", output_path)
