
# ============================================================
# AUTOMOTIVE NLP PROJECT
# PHASE 2: DATASET EXPLORATION
# ============================================================

import pandas as pd
import numpy as np
import json
import os

# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

file_path = "dataset/raw/automotive_dataset.jsonl"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Dataset not found at: {file_path}\n"
        "Check the dataset folder and filename."
    )

print("\nLoading Automotive Dataset...")

df = pd.read_json(file_path, lines=True)

print("Dataset loaded successfully!")


# ------------------------------------------------------------
# 2. BASIC DATASET OVERVIEW
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("1. DATASET OVERVIEW")
print("=" * 55)

print("Dataset shape:", df.shape)
print("Total records:", len(df))
print("Total columns:", len(df.columns))


# ------------------------------------------------------------
# 3. COLUMN NAMES
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("2. COLUMN NAMES")
print("=" * 55)

print(df.columns.tolist())


# ------------------------------------------------------------
# 4. DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("3. DATA TYPES")
print("=" * 55)

print(df.dtypes)


# ------------------------------------------------------------
# 5. FIRST FIVE RECORDS
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("4. FIRST FIVE RECORDS")
print("=" * 55)

print(df.head().to_string())


# ------------------------------------------------------------
# 6. DATASET INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("5. DATASET INFORMATION")
print("=" * 55)

df.info()


# ------------------------------------------------------------
# 7. MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("6. MISSING VALUES")
print("=" * 55)

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal missing values:", missing_values.sum())


# ------------------------------------------------------------
# 8. DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("7. DUPLICATE RECORDS")
print("=" * 55)

# Create a copy for duplicate checking
df_check = df.copy()

# Convert nested conversation lists into JSON strings
df_check["conversations"] = df_check["conversations"].apply(
    lambda x: json.dumps(
        x,
        ensure_ascii=False,
        sort_keys=True
    ) if isinstance(x, (list, dict)) else str(x)
)

# Check exact duplicate rows
exact_duplicates = df_check.duplicated().sum()

print("Exact duplicate rows:", exact_duplicates)

# Check duplicate instructions
duplicate_instructions = df["instruction"].duplicated().sum()

print("Duplicate instruction metadata:", duplicate_instructions)

# Check duplicate conversations
duplicate_conversations = (
    df_check["conversations"].duplicated().sum()
)

print("Duplicate conversations:", duplicate_conversations)


# ------------------------------------------------------------
# 9. LANGUAGE DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("8. LANGUAGE DISTRIBUTION")
print("=" * 55)

language_distribution = df["lang"].value_counts(dropna=False)

print(language_distribution)


# ------------------------------------------------------------
# 10. CONVERSATION STRUCTURE
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("9. CONVERSATION STRUCTURE")
print("=" * 55)

print("\nFirst conversation:")

first_conversation = df["conversations"].iloc[0]

print(first_conversation)

print("\nConversation roles:")

if isinstance(first_conversation, list):

    for message in first_conversation:

        print("\nRole:", message.get("from"))
        print("Content:", message.get("value", "")[:200])


# ------------------------------------------------------------
# 11. CONVERSATION TURN DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("10. CONVERSATION TURN DISTRIBUTION")
print("=" * 55)

df["conversation_turns"] = df["conversations"].apply(
    lambda x: len(x) if isinstance(x, list) else 0
)

print(df["conversation_turns"].value_counts().sort_index())


# ------------------------------------------------------------
# 12. MESSAGE ROLE DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("11. MESSAGE ROLE DISTRIBUTION")
print("=" * 55)

role_counts = {}

for conversation in df["conversations"]:

    if isinstance(conversation, list):

        for message in conversation:

            role = message.get("from", "unknown")

            role_counts[role] = role_counts.get(role, 0) + 1

print(pd.Series(role_counts, name="Message Count"))


# ------------------------------------------------------------
# 13. CONVERSATION LENGTH ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("12. CONVERSATION LENGTH ANALYSIS")
print("=" * 55)

df["text_length"] = df["conversations"].apply(
    lambda x: sum(
        len(str(message.get("value", "")))
        for message in x
        if isinstance(message, dict)
    ) if isinstance(x, list) else 0
)

print(df["text_length"].describe())


# ------------------------------------------------------------
# 14. DISPLAY SAMPLE ENGLISH RECORDS
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("13. SAMPLE ENGLISH RECORDS")
print("=" * 55)

english_df = df[df["lang"] == "en"]

print("Total English records:", len(english_df))

for index, row in english_df.head(3).iterrows():

    print("\nRecord ID:", row["id"])
    print("Instruction:", row["instruction"])
    print("Conversation:")

    print(row["conversations"])


# ------------------------------------------------------------
# 15. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("14. EXPLORATION SUMMARY")
print("=" * 55)

print("Total records:", len(df))
print("Total columns:", len(df.columns))
print("Missing values:", missing_values.sum())
print("Exact duplicate rows:", exact_duplicates)
print("Duplicate conversations:", duplicate_conversations)
print("English records:", len(english_df))
print("Chinese records:", len(df[df["lang"] == "zh"]))

print("\nDATASET EXPLORATION COMPLETED!")
print("=" * 55)
