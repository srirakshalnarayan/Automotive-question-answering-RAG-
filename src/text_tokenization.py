
# Import required libraries
import pandas as pd
import nltk

from nltk.tokenize import (
    sent_tokenize,
    word_tokenize,
    TreebankWordTokenizer,
    RegexpTokenizer
)

# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")


# ------------------------------------------------
# PART 1: SAMPLE AUTOMOTIVE PARAGRAPH
# ------------------------------------------------

paragraph = """
A car engine converts fuel into mechanical energy.
The engine requires air, fuel, and a spark to operate.
If the battery is weak, the vehicle may fail to start.
Regular maintenance improves vehicle performance.
"""


# ------------------------------------------------
# PART 2: SENTENCE TOKENIZATION
# ------------------------------------------------

print("\n========== SENTENCE TOKENIZATION ==========")

sentences = sent_tokenize(paragraph)

for i, sentence in enumerate(sentences, start=1):
    print(f"Sentence {i}: {sentence}")

print("Total sentences:", len(sentences))


# ------------------------------------------------
# PART 3: WORD TOKENIZATION
# ------------------------------------------------

print("\n========== WORD TOKENIZATION ==========")

words = word_tokenize(paragraph)

print(words)
print("Total tokens:", len(words))


# ------------------------------------------------
# PART 4: TREEBANK WORD TOKENIZATION
# ------------------------------------------------

print("\n========== TREEBANK TOKENIZATION ==========")

treebank_tokenizer = TreebankWordTokenizer()

treebank_words = treebank_tokenizer.tokenize(paragraph)

print(treebank_words)
print("Total tokens:", len(treebank_words))


# ------------------------------------------------
# PART 5: REGEXP TOKENIZATION
# ------------------------------------------------

print("\n========== REGEXP TOKENIZATION ==========")

# Extract words and numbers, ignoring punctuation
regexp_tokenizer = RegexpTokenizer(r"\w+")

regexp_words = regexp_tokenizer.tokenize(paragraph)

print(regexp_words)
print("Total tokens:", len(regexp_words))


# ------------------------------------------------
# PART 6: LOAD THE AUTOMOTIVE DATASET
# ------------------------------------------------

print("\n========== DATASET TOKENIZATION ==========")

df = pd.read_csv(
    "dataset/processed/automotive_cleaned_qa.csv",
    encoding="utf-8-sig"
)

print("Dataset shape:", df.shape)

# Select a few questions and answers for demonstration
sample_df = df.head(5)

for index, row in sample_df.iterrows():

    print(f"\nRecord {index + 1}")

    question = str(row["clean_question"])
    answer = str(row["clean_answer"])

    print("\nQuestion:", question)

    # Sentence tokenization
    question_sentences = sent_tokenize(question)
    print("Question sentences:", question_sentences)

    # Word tokenization
    question_words = word_tokenize(question)
    print("Question words:", question_words)

    # Treebank tokenization
    print(
        "Treebank tokens:",
        treebank_tokenizer.tokenize(question)
    )

    # Regexp tokenization
    print(
        "Regexp tokens:",
        regexp_tokenizer.tokenize(question)
    )

    print("\nAnswer:", answer)

    answer_sentences = sent_tokenize(answer)
    print("Answer sentences:", answer_sentences)

    answer_words = word_tokenize(answer)
    print("Answer words:", answer_words)


# ------------------------------------------------
# PART 7: SAVE TOKENIZATION RESULTS
# ------------------------------------------------

df["question_tokens"] = df["clean_question"].apply(
    lambda text: word_tokenize(str(text))
)

df["answer_tokens"] = df["clean_answer"].apply(
    lambda text: word_tokenize(str(text))
)

output_path = "dataset/processed/automotive_tokenized_qa.csv"

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nTokenization completed successfully!")
print("Tokenized dataset saved to:", output_path)
