
# Import required libraries
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger_eng")


# ------------------------------------------------
# PART 1: INITIALIZE STEMMERS AND LEMMATIZER
# ------------------------------------------------

porter = PorterStemmer()
snowball = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


# ------------------------------------------------
# PART 2: SAMPLE TEXT
# ------------------------------------------------

sample_text = """
Cars are running on roads.
The engines are performing efficiently.
Mechanics are repairing damaged vehicles.
Drivers were studying the vehicle manuals.
"""

tokens = word_tokenize(sample_text.lower())

# Keep only alphabetic words for this comparison
words = [word for word in tokens if word.isalpha()]

print("\n========== ORIGINAL WORDS ==========")
print(words)


# ------------------------------------------------
# PART 3: PORTER STEMMING
# ------------------------------------------------

print("\n========== PORTER STEMMING ==========")

porter_words = [porter.stem(word) for word in words]

print(porter_words)


# ------------------------------------------------
# PART 4: SNOWBALL STEMMING
# ------------------------------------------------

print("\n========== SNOWBALL STEMMING ==========")

snowball_words = [snowball.stem(word) for word in words]

print(snowball_words)


# ------------------------------------------------
# PART 5: LEMMATIZATION
# ------------------------------------------------

print("\n========== WORDNET LEMMATIZATION ==========")

lemma_words = [lemmatizer.lemmatize(word) for word in words]

print(lemma_words)


# ------------------------------------------------
# PART 6: POS-AWARE LEMMATIZATION
# ------------------------------------------------

print("\n========== POS-AWARE LEMMATIZATION ==========")

# Convert NLTK POS tags into WordNet POS tags
def get_wordnet_pos(tag):

    if tag.startswith("J"):
        return wordnet.ADJ

    elif tag.startswith("V"):
        return wordnet.VERB

    elif tag.startswith("N"):
        return wordnet.NOUN

    elif tag.startswith("R"):
        return wordnet.ADV

    else:
        return wordnet.NOUN


tagged_words = pos_tag(words)

pos_lemmas = []

for word, tag in tagged_words:

    pos = get_wordnet_pos(tag)

    lemma = lemmatizer.lemmatize(word, pos=pos)

    pos_lemmas.append(lemma)

print("\nPOS tagged words:")
print(tagged_words)

print("\nPOS-aware lemmas:")
print(pos_lemmas)


# ------------------------------------------------
# PART 7: COMPARISON USING A SMALL DATASET SAMPLE
# ------------------------------------------------

df = pd.read_csv(
    "dataset/processed/automotive_cleaned_qa.csv",
    encoding="utf-8-sig"
)

print("\nDataset shape:", df.shape)


def process_text(text):

    tokens = word_tokenize(str(text).lower())

    return [word for word in tokens if word.isalpha()]


sample_df = df.head(5)

for index, row in sample_df.iterrows():

    question = row["clean_question"]

    original_words = process_text(question)

    porter_result = [
        porter.stem(word) for word in original_words
    ]

    lemma_result = [
        lemmatizer.lemmatize(word) for word in original_words
    ]

    print(f"\nRecord {index + 1}")

    print("Original Question:", question)
    print("Original Tokens:", original_words)
    print("Porter Stemming:", porter_result)
    print("Lemmatization:", lemma_result)


# ------------------------------------------------
# PART 8: APPLY TECHNIQUES TO THE DATASET
# ------------------------------------------------

df["question_stemmed"] = df["clean_question"].apply(
    lambda text: [
        porter.stem(word) for word in process_text(text)
    ]
)

df["question_lemmatized"] = df["clean_question"].apply(
    lambda text: [
        lemmatizer.lemmatize(word) for word in process_text(text)
    ]
)

df["answer_stemmed"] = df["clean_answer"].apply(
    lambda text: [
        porter.stem(word) for word in process_text(text)
    ]
)

df["answer_lemmatized"] = df["clean_answer"].apply(
    lambda text: [
        lemmatizer.lemmatize(word) for word in process_text(text)
    ]
)


# ------------------------------------------------
# PART 9: SAVE RESULTS
# ------------------------------------------------

output_path = "dataset/processed/automotive_stemmed_lemmatized.csv"

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nStemming and lemmatization completed!")
print("Output saved to:", output_path)
