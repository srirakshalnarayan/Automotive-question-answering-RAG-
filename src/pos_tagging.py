
# Import required libraries
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.chunk import ne_chunk


# ------------------------------------------------
# PART 1: DOWNLOAD REQUIRED NLTK RESOURCES
# ------------------------------------------------

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")
nltk.download("tagsets_json")


# ------------------------------------------------
# PART 2: SAMPLE AUTOMOTIVE TEXT
# ------------------------------------------------

sample_text = """
Toyota manufactures reliable vehicles.
The car engine produces mechanical energy.
A mechanic repairs damaged engines carefully.
Electric vehicles use rechargeable batteries.
"""

print("\n========== SAMPLE AUTOMOTIVE TEXT ==========")
print(sample_text)


# ------------------------------------------------
# PART 3: WORD TOKENIZATION
# ------------------------------------------------

tokens = word_tokenize(sample_text)

print("\n========== WORD TOKENIZATION ==========")
print(tokens)

print("\nTotal tokens:", len(tokens))


# ------------------------------------------------
# PART 4: PART-OF-SPEECH TAGGING
# ------------------------------------------------

tagged_words = pos_tag(tokens)

print("\n========== POS TAGGING ==========")

for word, tag in tagged_words:
    print(f"{word:15} -> {tag}")


# ------------------------------------------------
# PART 5: DISPLAY POS TAG MEANINGS
# ------------------------------------------------

print("\n========== POS TAG MEANINGS ==========")

# Display each unique POS tag only once
unique_tags = sorted(set(tag for word, tag in tagged_words))

for tag in unique_tags:

    print(f"\nPOS Tag: {tag}")

    try:
        nltk.help.upenn_tagset(tag)

    except LookupError:
        print("POS description unavailable.")


# ------------------------------------------------
# PART 6: EXTRACT NOUNS, VERBS AND ADJECTIVES
# ------------------------------------------------

print("\n========== NOUNS, VERBS AND ADJECTIVES ==========")

nouns = [
    word for word, tag in tagged_words
    if tag.startswith("NN")
]

verbs = [
    word for word, tag in tagged_words
    if tag.startswith("VB")
]

adjectives = [
    word for word, tag in tagged_words
    if tag.startswith("JJ")
]

adverbs = [
    word for word, tag in tagged_words
    if tag.startswith("RB")
]

print("Nouns:", nouns)
print("Verbs:", verbs)
print("Adjectives:", adjectives)
print("Adverbs:", adverbs)


# ------------------------------------------------
# PART 7: NAMED ENTITY CHUNKING
# ------------------------------------------------

print("\n========== NAMED ENTITY CHUNKING ==========")

named_entities = ne_chunk(tagged_words)

print(named_entities)

# Extract named entities from the chunk tree
print("\nExtracted Named Entities:")

for subtree in named_entities:

    if hasattr(subtree, "label"):

        entity_name = " ".join(
            word for word, tag in subtree.leaves()
        )

        print(
            "Entity:", entity_name,
            "| Entity Type:", subtree.label()
        )


# ------------------------------------------------
# PART 8: LOAD AUTOMOTIVE DATASET
# ------------------------------------------------

input_path = "dataset/processed/automotive_cleaned_qa.csv"

df = pd.read_csv(
    input_path,
    encoding="utf-8-sig"
)

print("\n========== DATASET INFORMATION ==========")

print("Dataset shape:", df.shape)

print("\nDataset columns:")
print(df.columns.tolist())


# ------------------------------------------------
# PART 9: APPLY POS TAGGING TO DATASET
# ------------------------------------------------

def apply_pos_tagging(text):

    tokens = word_tokenize(str(text))

    tagged_tokens = pos_tag(tokens)

    return tagged_tokens


df["question_pos_tags"] = df["clean_question"].apply(
    apply_pos_tagging
)

df["answer_pos_tags"] = df["clean_answer"].apply(
    apply_pos_tagging
)


# ------------------------------------------------
# PART 10: DISPLAY DATASET POS TAGGING RESULTS
# ------------------------------------------------

print("\n========== DATASET POS TAGGING RESULTS ==========")

for index, row in df.head(5).iterrows():

    print(f"\nRecord {index + 1}")

    print("\nQuestion:")
    print(row["clean_question"])

    print("\nQuestion POS Tags:")
    print(row["question_pos_tags"])

    print("\nAnswer:")
    print(row["clean_answer"])

    print("\nAnswer POS Tags:")
    print(row["answer_pos_tags"])


# ------------------------------------------------
# PART 11: SAVE POS TAGGING RESULTS
# ------------------------------------------------

output_path = "dataset/processed/automotive_pos_tagged.csv"

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== PROCESS COMPLETED ==========")

print("POS tagging completed successfully!")

print("Final dataset shape:", df.shape)

print("Output saved to:", output_path)
