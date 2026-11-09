
# Import required libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------
# PART 1: LOAD AUTOMOTIVE DATASET
# ------------------------------------------------

input_path = "dataset/processed/automotive_cleaned_qa.csv"

df = pd.read_csv(
    input_path,
    encoding="utf-8-sig"
)

print("Original dataset shape:", df.shape)


# ------------------------------------------------
# PART 2: SELECT DATASET SAMPLE
# ------------------------------------------------

# Use 5000 records for the initial retrieval experiment
df = df.head(5000).copy()

df = df.dropna(
    subset=["clean_question", "clean_answer"]
)

df = df[
    (df["clean_question"].astype(str).str.strip() != "") &
    (df["clean_answer"].astype(str).str.strip() != "")
].copy()

df.reset_index(drop=True, inplace=True)

print("Records selected:", len(df))


# ------------------------------------------------
# PART 3: EXTRACT QUESTIONS
# ------------------------------------------------

questions = df["clean_question"].astype(str).tolist()

print("\n========== SAMPLE QUESTIONS ==========")

for question in questions[:5]:
    print("-", question)


# ------------------------------------------------
# PART 4: CREATE TF-IDF VECTORS
# ------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(questions)

print("\n========== TF-IDF INFORMATION ==========")

print("Number of questions:", tfidf_matrix.shape[0])
print("Vocabulary size:", tfidf_matrix.shape[1])
print("TF-IDF matrix shape:", tfidf_matrix.shape)


# ------------------------------------------------
# PART 5: DEFINE RETRIEVAL FUNCTION
# ------------------------------------------------

def retrieve_answers(user_query, top_k=5):

    # Convert user query into TF-IDF vector
    query_vector = vectorizer.transform([user_query])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # Get indices of top matching questions
    top_indices = similarity_scores.argsort()[::-1][:top_k]

    print("\n========== SEARCH RESULTS ==========")

    if len(top_indices) == 0 or similarity_scores[top_indices[0]] == 0:
        print("No relevant matching questions found.")
        return

    for rank, index in enumerate(top_indices, start=1):

        score = similarity_scores[index]

        # Skip results with no lexical overlap
        if score == 0:
            continue

        print(f"\nResult {rank}")
        print("Similarity Score:", round(score, 4))

        print("\nMatched Question:")
        print(df.iloc[index]["clean_question"])

        print("\nRetrieved Answer:")
        print(df.iloc[index]["clean_answer"])

        print("-" * 60)


# ------------------------------------------------
# PART 6: TEST WITH SAMPLE QUERIES
# ------------------------------------------------

sample_queries = [
    "Why is my car engine not starting?",
    "How can I maintain a vehicle battery?",
    "What causes engine overheating?"
]

print("\n========== SAMPLE QUERY TESTING ==========")

for query in sample_queries:

    print("\nUser Query:", query)

    retrieve_answers(query, top_k=3)


# ------------------------------------------------
# PART 7: INTERACTIVE QUESTION RETRIEVAL
# ------------------------------------------------

print("\n========== AUTOMOTIVE QUESTION SEARCH ==========")

print("Type 'exit' to stop the program.")

while True:

    user_query = input("\nEnter your automotive question: ").strip()

    if user_query.lower() == "exit":
        print("Automotive retrieval system closed.")
        break

    if not user_query:
        print("Please enter a question.")
        continue

    retrieve_answers(user_query, top_k=5)


# ------------------------------------------------
# PART 8: COMPLETION
# ------------------------------------------------

print("\nTF-IDF-based retrieval experiment completed!")
