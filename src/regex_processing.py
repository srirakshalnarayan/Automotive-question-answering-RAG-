
import pandas as pd
import re
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "dataset" / "processed" /
    "automotive_cleaned_qa.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "dataset" / "processed" /
    "automotive_regex_results.csv"
)

# ------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Original Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns.tolist())

# Identify question and answer columns
question_col = next(
    (col for col in df.columns
     if col.lower() in ["question", "instruction"]),
    None
)

answer_col = next(
    (col for col in df.columns
     if col.lower() in ["answer", "response", "output"]),
    None
)

if question_col is None or answer_col is None:
    raise ValueError(
        "Could not identify question and answer columns."
    )

# Work on a sample for demonstration
sample_df = df.head(5000).copy()

# ------------------------------------------------
# 2. REGEX 1: WHITESPACE NORMALIZATION
# ------------------------------------------------

def normalize_whitespace(text):
    text = str(text)

    # Replace multiple spaces, tabs and newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()


sample_df["cleaned_question"] = (
    sample_df[question_col].apply(normalize_whitespace)
)

sample_df["cleaned_answer"] = (
    sample_df[answer_col].apply(normalize_whitespace)
)

# ------------------------------------------------
# 3. REGEX 2: DIAGNOSTIC TROUBLE CODE EXTRACTION
# ------------------------------------------------

# Examples: P0300, P0420, B1234, C0035, U0100
dtc_pattern = r"\b[PCBU][0-9A-F][0-9A-F]{3}\b"

def extract_dtc_codes(text):
    return re.findall(
        dtc_pattern,
        str(text).upper()
    )

sample_df["diagnostic_codes"] = (
    sample_df["cleaned_question"].apply(extract_dtc_codes)
)

# ------------------------------------------------
# 4. REGEX 3: AUTOMOTIVE MEASUREMENT EXTRACTION
# ------------------------------------------------

measurement_pattern = (
    r"\b\d+(?:,\d{3})*(?:\.\d+)?"
    r"\s?(?:V|A|W|L|PSI|BAR|RPM|KM/H|MPH|KM|"
    r"MILES|MM|CM|°C|°F|C|F)\b"
    r"|\b\d+W-\d+\b"
)

def extract_measurements(text):
    return re.findall(
        measurement_pattern,
        str(text),
        flags=re.IGNORECASE
    )

sample_df["automotive_measurements"] = (
    sample_df["cleaned_question"].apply(
        extract_measurements
    )
)

# ------------------------------------------------
# 5. DISPLAY RESULTS
# ------------------------------------------------

print("\n--- REGEX PROCESSING RESULTS ---")

for index, row in sample_df.head(10).iterrows():

    print("\nQuestion:", row[question_col])

    print("Normalized:", row["cleaned_question"])

    print("Diagnostic Codes:", row["diagnostic_codes"])

    print(
        "Measurements:",
        row["automotive_measurements"]
    )

# ------------------------------------------------
# 6. SAVE RESULTS
# ------------------------------------------------

sample_df.to_csv(OUTPUT_FILE, index=False)

print("\nRegex processing completed!")
print("Records processed:", len(sample_df))
print("Saved to:", OUTPUT_FILE)
