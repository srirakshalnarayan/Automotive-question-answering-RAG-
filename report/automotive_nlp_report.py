
"""
============================================================
AUTOMOTIVE QUESTION UNDERSTANDING AND ANSWER GENERATION
USING NATURAL LANGUAGE PROCESSING AND
RETRIEVAL-AUGMENTED GENERATION
============================================================

Student Name       : Sriraksha L
Registration Number: 2539756
Programme          : M.Sc. Artificial Intelligence and Cybersecurity

============================================================
1. PROBLEM DEFINITION
============================================================

1.1 INTRODUCTION

The automobile industry involves a wide range of technical
information, covering areas such as engine performance,
vehicle components, safety systems, maintenance, manufacturing
processes and emerging automotive technologies. With the
continuous development of the industry, the amount of
information available in textual form has also increased.

Although this information is useful, finding specific and
relevant answers from a large collection of automotive content
can sometimes be difficult. For instance, a user might ask a
question about engine overheating, vehicle safety regulations
or the functioning of an automotive component. Finding an
appropriate answer requires more than simply matching
individual keywords, as users may express the same question
in different ways.

Natural Language Processing (NLP) provides techniques that
help computers process, analyse and understand human language.
By applying these techniques, textual automotive information
can be cleaned, transformed and represented in a format
suitable for information retrieval and further analysis.

This project focuses on developing an automotive
question-understanding and answer-generation system that
combines NLP techniques with Retrieval-Augmented Generation
(RAG). The system accepts a natural language question,
identifies relevant information from an automotive
question-answer dataset and uses a language model to generate
a contextual response.


1.2 PROBLEM STATEMENT

Retrieving accurate and relevant information from a large
automotive text collection can be challenging when a user's
question does not contain the exact words or phrases present
in the available content.

Conventional keyword-based approaches may not always
recognise the underlying meaning of differently phrased
questions.

The problem addressed in this project is the development of
an NLP-based system that can process automotive-related
questions, identify semantically relevant question-answer
records and generate meaningful responses using the
retrieved information as context.

The proposed solution uses text preprocessing, feature
extraction, numerical text representation, sentence
embeddings, similarity-based retrieval and language
model-based answer generation.


1.3 OBJECTIVES

The main objectives of this project are:

1. To explore and prepare an automotive question-answer
   dataset for NLP applications.

2. To apply text preprocessing techniques such as
   tokenization, text normalization, stop-word removal,
   stemming and lemmatization.

3. To use Regular Expressions (Regex) for identifying and
   processing specific automotive-related text patterns.

4. To represent textual data using Bag-of-Words (BoW)
   and Term Frequency-Inverse Document Frequency (TF-IDF).

5. To explore semantic representations using sentence
   embeddings and compare them with traditional
   TF-IDF-based retrieval.

6. To develop a Retrieval-Augmented Generation (RAG)
   pipeline that retrieves relevant automotive information
   and generates contextual answers.


1.4 RELEVANCE OF THE PROBLEM

Automotive information is relevant to vehicle owners,
students, automotive enthusiasts and individuals interested
in vehicle engineering and technology.

A system that allows users to ask questions in natural
language can make such information easier to access without
requiring them to search through large collections of text
manually.

This project also demonstrates the practical application
of foundational NLP concepts. Techniques such as tokenization,
normalization, feature extraction and text vectorization
form the basis for more advanced applications, including
semantic search and question-answering systems.

By combining traditional NLP methods with modern retrieval
and language generation techniques, the project illustrates
how unstructured text can be transformed into a useful
interactive information retrieval application.


============================================================
2. DATASET DESCRIPTION AND PREPARATION
============================================================

2.1 DATASET SOURCE

The dataset selected for this project is
IndustryInstruction_Automobiles, published by the Beijing
Academy of Artificial Intelligence (BAAI) and hosted on
Hugging Face.

Dataset link:
https://huggingface.co/datasets/BAAI/IndustryInstruction_Automobiles

The dataset was selected because it contains
instruction-based conversations related to the automobile
industry. Its question-answer format makes it suitable
for exploring text preprocessing, language understanding
and information retrieval techniques.


2.2 DATASET OVERVIEW

The original dataset contains 126,707 records, covering
automotive-related instructions and responses in both
English and Chinese.

It includes the following seven fields:

1. deita_score
2. rw_score
3. id
4. instruction
5. lang
6. length
7. conversations

An initial exploration of the dataset showed that it
contained 48,671 English records and 78,036 Chinese records.

Since the project focuses on English-language NLP processing,
the English records were selected for further analysis.


2.3 DATASET EXTRACTION AND PREPROCESSING

The original dataset was provided in JSONL (JSON Lines)
format. It was first loaded and explored to understand
its structure, attributes and language distribution.

The English-language conversations were then extracted,
and the corresponding questions and answers were organised
into a structured CSV file.

This produced an English question-answer dataset containing
48,671 records.

Further cleaning operations were performed to improve
the consistency of the text before applying NLP techniques.
These included:

- Removing duplicate question-answer records.
- Normalising text using Unicode normalization.
- Removing unnecessary control characters.
- Cleaning unwanted whitespace and formatting
  inconsistencies.
- Organising the cleaned questions and answers into
  separate columns.

The final cleaned dataset was saved as
automotive_cleaned_qa.csv and used for subsequent
NLP processing and retrieval experiments.


2.4 DATASET SUITABILITY

The dataset provides a collection of automotive-related
questions and corresponding responses, making it suitable
for studying how NLP techniques can be used to process
and retrieve information from domain-specific text.

It also supports the development of a question-answering
pipeline in which user queries can be compared with
existing automotive questions to identify relevant
information.

The dataset is an instruction-based automotive text
collection rather than a certified vehicle repair manual.
Therefore, generated responses are intended for general
automotive information retrieval and should not be treated
as verified mechanical or diagnostic instructions.


============================================================
3. TEXT PREPROCESSING
============================================================

Text preprocessing is an important stage in Natural
Language Processing because raw text often contains
unnecessary characters, inconsistent formatting and
different forms of words.

In this project, preprocessing was performed on the
extracted automotive questions and answers to improve
text consistency and prepare the data for further
NLP operations.

The following techniques were implemented:


3.1 TEXT NORMALIZATION

Text normalization was performed to maintain consistency
across the dataset.

Unicode normalization using NFKC was applied to standardise
text representations. Unnecessary control characters,
extra spaces and formatting inconsistencies were also
handled.

This step helps reduce variations in the text before
tokenization and feature extraction.


3.2 TOKENIZATION

Tokenization is the process of splitting text into smaller
units called tokens.

Two main approaches were explored:

- Sentence Tokenization: Divides a paragraph into
  individual sentences.

- Word Tokenization: Splits sentences into individual
  words or tokens.

NLTK tokenization methods were used along with
TreebankWordTokenizer and RegexpTokenizer to examine
different tokenization outputs.

Tokenization converts unstructured text into smaller
units that can be processed and analysed further.


3.3 STOP-WORD REMOVAL

Stop-word removal involves identifying and removing
frequently occurring words that generally contribute
less information to a particular text analysis task.

The NLTK English stop-word list was used for this process.

However, important negation words such as "not", "no",
"nor" and "never" were preserved because they can change
the meaning of automotive statements.

Preserving negation is important because removing the
word "not" could change the meaning of a statement.


3.4 STEMMING

Stemming is a text normalization technique that reduces
words to their root or base-like form by removing
common prefixes or suffixes.

Porter Stemmer and Snowball Stemmer from NLTK were
implemented and their outputs were compared.

Stemming helps reduce vocabulary variations during
text processing. However, the resulting stem may not
always be a valid dictionary word.


3.5 LEMMATIZATION

Lemmatization reduces words to their meaningful dictionary
base form, known as a lemma.

WordNetLemmatizer was used to perform lemmatization
with the help of the WordNet lexical database.

Unlike stemming, lemmatization considers linguistic
information to produce meaningful base forms.

This helps group different forms of a word under a
common meaningful representation.


3.6 SUMMARY OF PREPROCESSING

The preprocessing stage transformed the extracted
automotive text into a cleaner and more consistent form.

Tokenization divided text into manageable units,
stop-word removal reduced less informative terms,
and stemming and lemmatization helped handle variations
in word forms.

These operations prepared the dataset for subsequent
text representation techniques such as Bag-of-Words
and TF-IDF.

# ============================================================
# 4. REGULAR EXPRESSIONS
# ============================================================

Regular Expressions (Regex) are patterns used to identify,
extract, match or replace specific sequences of characters
within text.

In this project, regular expressions were used to process
automotive-related text and identify useful patterns that
may not be handled through ordinary text cleaning alone.

The regex processing was implemented using Python's re
module.


4.1 WHITESPACE NORMALIZATION

Unnecessary spaces, tabs and newline characters can make
text inconsistent. A regular expression was used to identify
multiple consecutive whitespace characters and replace
them with a single space.

Pattern:
r'\\s+'

Example:

Input:
"The    engine     needs
regular maintenance."

Output:
"The engine needs regular maintenance."

This helps maintain consistent formatting across the dataset.


4.2 DIAGNOSTIC TROUBLE CODE EXTRACTION

Diagnostic Trouble Codes (DTCs) are alphanumeric codes
used in vehicle diagnostic systems to identify potential
faults.

A regular expression was used to identify automotive
diagnostic code patterns.

Example pattern:
r'\\b[PCBU][0-9A-Fa-f]{4}\\b'

Example matches:
P0301
P0420
B0020
C1234

This pattern identifies codes beginning with P, C, B or U,
followed by four hexadecimal characters.

It helps identify diagnostic code references within
automotive text.


4.3 AUTOMOTIVE MEASUREMENT EXTRACTION

Automotive text frequently contains measurements such as
engine displacement, speed, temperature and pressure.

Regular expressions were used to identify measurement
values and their associated units.

Examples of automotive measurements include:

- 2.0 L
- 150 km/h
- 90 °C
- 35 psi

This helps identify and extract useful technical
measurement information from unstructured automotive text.


4.4 IMPORTANCE OF REGEX IN THE PROJECT

Regular expressions helped identify specific patterns
within automotive text and improve text cleaning.

Whitespace normalization improves consistency, while
diagnostic code and measurement extraction help identify
automotive-specific information.

These operations complement the other preprocessing
techniques used in the project.


# ============================================================
# 5. TEXT REPRESENTATION
# ============================================================

Text representation is the process of converting textual
information into numerical form so that machine learning
and NLP algorithms can process it.

Since computers cannot directly perform mathematical
operations on ordinary sentences, text must be represented
as numerical features.

In this project, two traditional text representation
techniques were implemented: Bag-of-Words and TF-IDF.


5.1 BAG-OF-WORDS (BoW)

Bag-of-Words is a text representation technique that
converts documents into numerical vectors based on the
frequency of words appearing in them.

Each unique word is treated as a feature, and its
frequency is recorded in the corresponding document.

CountVectorizer from Scikit-learn was used to implement
Bag-of-Words on the automotive questions.

The technique was applied to a sample of 5,000 questions.

For example:

Document 1:
"The car engine needs maintenance."

Document 2:
"The engine requires regular maintenance."

The vocabulary is created from the unique words in
both documents. Each document is then represented
using the frequency of these words.

The resulting matrix contains numerical values
representing word occurrences.

INTERPRETATION:

Bag-of-Words provides a simple way to represent text
using word frequencies. It is useful for identifying
frequently occurring terms, but it does not consider
word order or the contextual meaning of words.


5.2 TERM FREQUENCY-INVERSE DOCUMENT FREQUENCY (TF-IDF)

TF-IDF is a text representation technique that measures
the importance of a word within a document relative
to a collection of documents.

It combines two measures:

Term Frequency (TF):
Measures how frequently a word appears in a document.

Inverse Document Frequency (IDF):
Measures how uncommon or informative a word is across
the document collection.

A commonly used representation is:

TF-IDF = TF × IDF

TfidfVectorizer from Scikit-learn was used to generate
TF-IDF representations of the automotive questions.

The technique was applied to a sample of 5,000 questions.

INTERPRETATION:

Unlike Bag-of-Words, TF-IDF gives greater importance
to terms that are relatively distinctive across the
document collection.

For example, a technical term such as "turbocharger"
may receive a higher weight than a very common word
because it provides more information about the subject.

TF-IDF is useful for identifying important terms and
supporting keyword-based document retrieval.


5.3 COMPARISON OF BoW AND TF-IDF

Bag-of-Words and TF-IDF both convert text into numerical
representations, but they differ in how they assign
importance to words.

Bag-of-Words records the frequency of each word, whereas
TF-IDF considers both the frequency of a word in a
document and its distribution across the document
collection.

Both techniques were explored as part of the project
to understand traditional approaches to text
representation before moving towards semantic
sentence embeddings.


# ============================================================
# 6. COMPLETE NLP PIPELINE
# ============================================================

The complete project combines traditional NLP techniques
with semantic information retrieval and language model-based
answer generation.

The pipeline begins with raw automotive question-answer
data and ends with an AI-generated response to a user's
question.


6.1 DATA LOADING AND EXPLORATION

The original automotive dataset was loaded in JSONL format.
Its structure, attributes and language distribution were
examined to understand the available information.

English-language question-answer records were extracted
for further processing.


6.2 TEXT PREPROCESSING

The extracted text was cleaned and normalised.

Tokenization, stop-word removal, stemming, lemmatization
and regular expression processing were implemented to
prepare the text for subsequent NLP operations.


6.3 TEXT REPRESENTATION

Bag-of-Words and TF-IDF were implemented to understand
traditional numerical representations of text.

Sentence embeddings were also generated using the
all-MiniLM-L6-v2 model.

These embeddings represent the semantic information
in automotive questions as numerical vectors.


6.4 SEMANTIC RETRIEVAL USING FAISS

The generated sentence embeddings were stored in a
FAISS vector index.

FAISS (Facebook AI Similarity Search) was used to retrieve
automotive questions that were semantically similar
to a user's input query.

The system retrieves the top 20 candidate records
based on vector similarity.


6.5 CROSS-ENCODER RERANKING

The retrieved candidates are further evaluated using
a cross-encoder model.

The model used in the project is:

cross-encoder/ms-marco-MiniLM-L-6-v2

It evaluates the relationship between the user's query
and each retrieved question.

The candidates are then sorted based on their
reranking scores, and the top three records are selected.


6.6 CONTEXT PREPARATION

The questions and answers from the selected records
are combined into a structured context.

This context provides relevant automotive information
to the language model for answer generation.


6.7 ANSWER GENERATION USING QWEN AND OLLAMA

The retrieved context and user's question are passed
to the Qwen3.5 4B language model through the Ollama
runtime.

The model generates a response using the retrieved
information as its primary context.

Streaming is enabled so that the generated response
is displayed progressively in the Streamlit application.

The system also instructs the model not to invent facts
and to indicate when the retrieved information is
insufficient.


6.8 USER INTERFACE

A Streamlit-based application was developed to provide
an interactive interface.

Users can enter automotive-related questions, view
AI-generated answers, inspect the retrieved sources
and download the generated response.

The application combines the different stages of the
NLP and RAG pipeline into a single interface.


6.9 PIPELINE SUMMARY

Raw Automotive Dataset
        |
        v
Dataset Exploration
        |
        v
English Question-Answer Extraction
        |
        v
Text Cleaning and Preprocessing
        |
        v
Tokenization, Stop-word Removal,
Stemming and Lemmatization
        |
        v
Regex Processing
        |
        v
BoW and TF-IDF Representation
        |
        v
Sentence Embedding Generation
        |
        v
FAISS Similarity Retrieval
        |
        v
Cross-Encoder Reranking
        |
        v
Context Preparation
        |
        v
Qwen Answer Generation
        |
        v
Streamlit Application Output


# ============================================================
# 7. CONCLUSION
# ============================================================

This project demonstrates the application of Natural
Language Processing techniques to an automotive
question-answering problem.

The work began with the exploration and preparation
of an automotive instruction-based dataset. Text
preprocessing techniques such as normalization,
tokenization, stop-word removal, stemming and
lemmatization were implemented to improve the
consistency of the textual data.

Regular Expressions were used to identify and process
automotive-related patterns, including whitespace,
diagnostic trouble codes and measurements.

Bag-of-Words and TF-IDF were explored to understand
traditional numerical representations of text.
Sentence embeddings were then used to represent the
semantic information contained in automotive questions.

These embeddings were integrated with FAISS for
similarity-based retrieval, followed by cross-encoder
reranking to select relevant records.

Finally, Retrieval-Augmented Generation was implemented
using Qwen3.5 4B through Ollama to generate contextual
answers within a Streamlit application.

The project provided practical experience in combining
foundational NLP techniques with modern semantic
retrieval and language generation methods.

One limitation is that the quality of the generated
answer depends on the relevance and completeness of
the retrieved information. The dataset is not a
certified automotive repair or diagnostic reference,
so generated responses should be treated as general
informational assistance rather than verified
technical instructions.


# ============================================================
# END OF REPORT
# ============================================================
"""

