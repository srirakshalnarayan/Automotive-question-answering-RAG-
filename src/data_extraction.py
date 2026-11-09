
import pandas as pd
import os

# ------------------------------------------------
# 1. LOAD ORIGINAL DATASET
# ------------------------------------------------

file_path = "dataset/raw/automotive_dataset.jsonl"

df = pd.read_json(file_path, lines=True)

print("Original dataset loaded:", df.shape)


# ------------------------------------------------
# 2. FILTER ENGLISH RECORDS
# ------------------------------------------------

english_df = df[df["lang"] == "en"].copy()

print("\nEnglish records:", len(english_df))


# ------------------------------------------------
# 3. EXTRACT QUESTIONS AND ANSWERS
# ------------------------------------------------

extracted_records = []

unmatched_records = 0

for index, row in english_df.iterrows():

    conversation = row["conversations"]

    if not isinstance(conversation, list):
        unmatched_records += 1
        continue

    pending_question = None

    for message in conversation:

        if not isinstance(message, dict):
            continue

        role = message.get("from")
        text = message.get("value", "")

        if not isinstance(text, str) or not text.strip():
            continue

        if role == "human":

            pending_question = text.strip()

        elif role == "gpt" and pending_question is not None:

            extracted_records.append({

                "record_index": index,

                "instruction_type": row["id"],

                "question": pending_question,

                "answer": text.strip(),

                "deita_score": row["deita_score"],

                "rw_score": row["rw_score"],

                "length": row["length"]

            })

            pending_question = None

    if pending_question is not None:
        unmatched_records += 1


# ------------------------------------------------
# 4. CREATE STRUCTURED DATAFRAME
# ------------------------------------------------

qa_df = pd.DataFrame(extracted_records)

print("\nExtracted question-answer records:", len(qa_df))

print("Unmatched records:", unmatched_records)


# ------------------------------------------------
# 5. CHECK EXTRACTED DATA
# ------------------------------------------------

print("\n========== EXTRACTED DATA ==========")

print("\nDataset shape:")
print(qa_df.shape)

print("\nColumn names:")
print(qa_df.columns.tolist())

print("\nFirst five question-answer pairs:")
print(qa_df[["question", "answer"]].head().to_string())


# ------------------------------------------------
# 6. CHECK MISSING VALUES
# ------------------------------------------------

print("\n========== MISSING VALUES ==========")

print(qa_df.isnull().sum())


# ------------------------------------------------
# 7. SAVE EXTRACTED DATASET
# ------------------------------------------------

output_dir = "dataset/processed"

os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "automotive_english_qa.csv"
)

qa_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\nExtracted dataset saved successfully!")
print("Location:", output_path)

print("\nDATA EXTRACTION COMPLETED!")
