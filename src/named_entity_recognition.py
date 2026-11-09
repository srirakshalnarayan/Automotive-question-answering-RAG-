
# Import required libraries
import pandas as pd
import spacy
from collections import Counter


# ------------------------------------------------
# PART 1: LOAD SPACY MODEL
# ------------------------------------------------

nlp = spacy.load("en_core_web_sm")

print("spaCy model loaded successfully!")


# ------------------------------------------------
# PART 2: SAMPLE AUTOMOTIVE TEXT
# ------------------------------------------------

sample_text = """
Toyota is a Japanese automobile manufacturer.
BMW manufactures luxury cars in Germany.
Hyundai produces electric vehicles.
The Tesla Model 3 uses a lithium-ion battery.
"""

print("\n========== SAMPLE TEXT ==========")
print(sample_text)


# ------------------------------------------------
# PART 3: NAMED ENTITY RECOGNITION
# ------------------------------------------------

doc = nlp(sample_text)

print("\n========== NAMED ENTITIES ==========")

for entity in doc.ents:
    print(
        entity.text,
        "->",
        entity.label_,
        "->",
        spacy.explain(entity.label_)
    )


# ------------------------------------------------
# PART 4: LOAD AUTOMOTIVE DATASET
# ------------------------------------------------

input_path = "dataset/processed/automotive_cleaned_qa.csv"

df = pd.read_csv(input_path, encoding="utf-8-sig")

print("\nOriginal dataset shape:", df.shape)


# ------------------------------------------------
# PART 5: SELECT A SAMPLE OF RECORDS
# ------------------------------------------------

# Process only 1000 records to reduce execution time
sample_df = df.head(1000).copy()

print("Records selected for NER:", len(sample_df))


# ------------------------------------------------
# PART 6: BATCH ENTITY EXTRACTION
# ------------------------------------------------

def extract_entities_batch(texts):

    results = []

    # nlp.pipe processes multiple texts efficiently
    for doc in nlp.pipe(texts, batch_size=100):

        entities = []

        for entity in doc.ents:

            entities.append({
                "entity": entity.text,
                "label": entity.label_
            })

        results.append(entities)

    return results


# Process questions and answers in batches
sample_df["question_entities"] = extract_entities_batch(
    sample_df["clean_question"].fillna("").astype(str)
)

sample_df["answer_entities"] = extract_entities_batch(
    sample_df["clean_answer"].fillna("").astype(str)
)


# ------------------------------------------------
# PART 7: DISPLAY RESULTS
# ------------------------------------------------

print("\n========== DATASET NER RESULTS ==========")

for index, row in sample_df.head(10).iterrows():

    print(f"\nRecord {index + 1}")

    print("Question:", row["clean_question"])
    print("Question Entities:", row["question_entities"])

    print("Answer:", row["clean_answer"])
    print("Answer Entities:", row["answer_entities"])


# ------------------------------------------------
# PART 8: ENTITY FREQUENCY ANALYSIS
# ------------------------------------------------

print("\n========== ENTITY FREQUENCY ANALYSIS ==========")

entity_counts = Counter()

for entities in sample_df["question_entities"]:

    for entity in entities:

        entity_counts[entity["label"]] += 1


for label, count in entity_counts.most_common():

    print(f"{label}: {count}")


# ------------------------------------------------
# PART 9: SAVE RESULTS
# ------------------------------------------------

output_path = "dataset/processed/automotive_ner_results.csv"

sample_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nNER completed successfully!")
print("Records processed:", len(sample_df))
print("Output saved to:", output_path)
