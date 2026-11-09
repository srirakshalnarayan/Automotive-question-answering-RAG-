
from pathlib import Path

# ------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CONTEXT_FILE = (
    BASE_DIR / "dataset" / "processed" /
    "last_rag_context.txt"
)

OUTPUT_FILE = (
    BASE_DIR / "dataset" / "processed" /
    "rag_prompt.txt"
)

# ------------------------------------------------
# 2. READ RETRIEVED CONTEXT
# ------------------------------------------------

def load_context():

    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            "Retrieved context not found. "
            "Run prepare_rag_context.py first."
        )

    context = CONTEXT_FILE.read_text(
        encoding="utf-8"
    ).strip()

    if not context:
        raise ValueError("Retrieved context is empty.")

    return context

# ------------------------------------------------
# 3. CONSTRUCT RAG PROMPT
# ------------------------------------------------

def build_prompt(question, context):

    prompt = f"""
You are an automotive question-answering assistant.

Your task is to answer the user's question using
the retrieved information provided below.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:

1. Use the retrieved context as the primary source
   for answering the question.

2. Provide a clear, accurate and understandable answer.

3. Do not invent facts or unsupported technical details.

4. If the retrieved context does not contain enough
   information, clearly state that the available
   information is insufficient.

5. If multiple sources provide relevant information,
   combine them into a coherent response.

6. Do not claim that a vehicle has a particular fault
   without sufficient evidence.

7. Keep the answer relevant to the user's question.

FINAL ANSWER:
"""

    return prompt.strip()

# ------------------------------------------------
# 4. MAIN PROGRAM
# ------------------------------------------------

if __name__ == "__main__":

    question = input(
        "\nEnter your automotive question: "
    ).strip()

    if not question:
        raise ValueError("Please enter a question.")

    context = load_context()

    final_prompt = build_prompt(
        question,
        context
    )

    # Save prompt
    OUTPUT_FILE.write_text(
        final_prompt,
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("GENERATED RAG PROMPT")
    print("=" * 70)

    print(final_prompt)

    print("\n" + "=" * 70)
    print("Prompt construction completed!")
    print("Saved to:", OUTPUT_FILE)
