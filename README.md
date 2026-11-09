# Automotive Question Understanding and Answer Generation Using NLP and RAG

Sriraksha L   

## 1. Project Overview

This project is an automotive question-answering application. It combines natural language processing (NLP), semantic retrieval and retrieval-augmented generation (RAG) to retrieve relevant automotive question-and-answer information and generate a response to a user's query.

The application uses a Streamlit interface. Sentence embeddings and a FAISS vector index retrieve relevant questions. A cross-encoder reranks the retrieved results, and the selected information is provided as context to a locally running Qwen model through Ollama.

This is an informational prototype, not a certified vehicle repair, diagnostic or safety system.

## 2. Technologies Used

- Python 3.12
- Pandas and NumPy for data handling
- NLTK and spaCy for NLP processing
- scikit-learn for text representation and machine-learning utilities
- Sentence Transformers for embeddings and reranking
- FAISS for vector similarity retrieval
- Ollama with `qwen3.5:4b` for local answer generation
- Streamlit for the application interface

## 3. Project Structure

```text
automotive_rag_nlp/
├── app/
│   └── app.py
├── dataset/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── data_exploration.py
│   ├── data_extraction.py
│   ├── data_preprocessing.py
│   ├── text_tokenization.py
│   ├── stopword_removal.py
│   ├── stemming_lemmatization.py
│   ├── pos_tagging.py
│   ├── named_entity_recognition.py
│   ├── ngram_analysis.py
│   ├── bag_of_words.py
│   ├── tfidf_vectorization.py
│   ├── tfidf_retrieval.py
│   ├── word2vec_embeddings.py
│   ├── sentence_embeddings.py
│   ├── retrieval_comparison.py
│   ├── build_vector_store.py
│   ├── rag_retrieval.py
│   ├── reranked_retrieval.py
│   ├── regex_processing.py
│   ├── prepare_rag_context.py
│   ├── build_rag_prompt.py
│   ├── retrieval_relevance_test.py
│   └── automotive_rag_pipeline.py
├── vector_store/
│   ├── automotive_faiss.index
│   └── automotive_metadata.csv
├── README.md
└── requirements.txt
```

The dataset and vector-store files should remain in their existing locations. Generated analysis files may vary depending on which scripts have been run.

## 4. Environment Setup

Use Python 3.12. From the project root, create and activate a virtual environment.

### Windows PowerShell

```powershell
py -3.12 -m venv venv312
.\venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If an NLP script reports a missing NLTK resource, download the specific resource named in the error using `nltk.download()`.

## 5. Ollama Setup

Install Ollama separately and ensure it is running. Check whether the configured model is available:

```powershell
ollama list
```

The application uses:

```text
qwen3.5:4b
```

If it is not listed, download it:

```powershell
ollama pull qwen3.5:4b
```

Ollama is a separate application and is not installed through `requirements.txt`.

## 6. Run the Application

From the project root, with the virtual environment activated, run:

```powershell
streamlit run app/app.py
```

Open the local URL displayed in the terminal if the browser does not open automatically.

## 7. Processing and Retrieval Workflow

1. The automotive dataset is explored and English question-and-answer records are extracted.
2. Text is cleaned and normalised.
3. NLP processing includes tokenisation, stop-word removal, stemming, lemmatisation, POS tagging, named entity recognition and n-gram analysis.
4. Bag of Words and TF-IDF representations are generated for text analysis.
5. Sentence embeddings are created for the question records.
6. FAISS retrieves the most similar candidate questions.
7. A cross-encoder reranks the retrieved candidates.
8. Relevant question-and-answer content is prepared as context.
9. The user query and retrieved context are sent to Qwen through Ollama.
10. The generated response is displayed in the Streamlit application with retrieved source information.

## 8. Important Notes

- Run Streamlit from the project root so relative file paths resolve correctly.
- Keep the FAISS index and its corresponding metadata together. If either is moved or replaced, update the paths in the application.
- Embedding models may need to be downloaded the first time they are loaded.
- Retrieval similarity does not guarantee that a generated answer is correct. Review the retrieved sources and independently verify important automotive information.
- The small retrieval relevance test is a basic functional check, not a general measure of system accuracy.
- Do not use this prototype as a substitute for a qualified mechanic, official vehicle documentation or emergency assistance.

## 9. Application Run Command

```powershell
streamlit run app/app.py
```
