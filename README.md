# Chat With Your Documents — RAG Project

A local web app that lets users upload PDF, DOCX, TXT, and Markdown files and ask questions grounded in those documents.

## Architecture

`Document → Text extraction → Chunking → Embeddings → FAISS vector index → Semantic retrieval → LLM → Answer with source references`

## Features

- Upload multiple PDF, DOCX, TXT, and Markdown documents.
- Extract and clean document text.
- Split text into overlapping chunks (900 words, 150-word overlap by default).
- Generate local embeddings with `all-MiniLM-L6-v2`.
- Store normalized vectors in a persistent FAISS index.
- Retrieve the top-k semantically similar chunks.
- Send retrieved context to an OpenAI chat model.
- Prompt the model to answer only from retrieved context and admit when evidence is missing.
- Display source file names, chunk numbers, and similarity scores.

## Requirements

- Python 3.10 or newer
- Internet connection for first-time package/model installation
- An OpenAI API key for answer generation (embeddings are local)

## Run locally

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Open .env and set OPENAI_API_KEY
streamlit run app.py
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
streamlit run app.py
```

Open the local URL shown by Streamlit (normally http://localhost:8501).

## How to use

1. Upload one or more supported files in the sidebar.
2. Select **Process documents** and wait for indexing to finish.
3. Enter your OpenAI API key if it is not already configured.
4. Ask a question in the chat box.
5. Inspect the retrieved passages and source references to verify the answer.

The FAISS index and chunk metadata are stored in `vector_store/`. Delete that folder to clear the local document index.

## Key concepts

### Hallucinations
An LLM hallucination is an answer that sounds plausible but is false, unsupported, or fabricated. Models generate likely text from learned patterns; they do not automatically verify every claim against trusted evidence. Hallucinations can be reduced—but not eliminated—by grounding answers in trusted sources, asking the model to abstain when evidence is missing, and showing citations.

### Retrieval-Augmented Generation (RAG)
RAG combines information retrieval with text generation. Instead of asking an LLM to answer from its parameters alone, the application first searches an external knowledge collection, then supplies relevant passages as context for the answer.

### Embeddings and semantic similarity
An embedding is a numeric vector representing the meaning of text. Texts with similar meanings tend to have vectors close together in the embedding space. This enables semantic search even when the query and document use different words.

### Chunking
Long documents are divided into smaller passages. Chunking helps retrieval focus on relevant sections and keeps the prompt context manageable. Overlap preserves information that crosses chunk boundaries. Chunk size and overlap should be tested against the document type.

### Vector database
A vector database/index stores embeddings and supports nearest-neighbor similarity search. This project uses FAISS with cosine similarity implemented as inner product over normalized vectors.

## Improving retrieval quality

- Tune chunk size and overlap; compare 400, 700, and 900-word chunks.
- Adjust top-k and inspect whether the correct evidence is retrieved.
- Add metadata filters (document, date, section) for larger collections.
- Use a reranker after initial vector retrieval.
- Consider hybrid search (keyword/BM25 + vector search).
- Remove repeated headers, footers, and irrelevant boilerplate.
- Evaluate with a fixed set of questions and expected source passages.

## Limitations and safety

- Retrieval quality depends on text extraction and chunk boundaries.
- Scanned PDFs need OCR; this starter app extracts selectable PDF text only.
- Similarity scores are relevance signals, not probabilities of correctness.
- RAG reduces unsupported answers but does not guarantee factual accuracy.
- Do not upload confidential or sensitive documents unless you are authorized to process them.
- The app sends the retrieved text and question to the OpenAI API for answer generation.

## Suggested evaluation

Create 10–20 test questions from your own documents. For each, record:
1. Whether the correct chunk appears in the top-k results (Recall@k).
2. Whether the answer is supported by the retrieved passages.
3. Whether the cited source is correct.
4. Whether the model correctly says “not found” for unanswerable questions.

## Deliverables checklist

- [x] RAG concepts and hallucination explanation
- [x] Document extraction and cleaning
- [x] Text chunking with overlap
- [x] Embedding generation
- [x] Persistent vector index
- [x] Semantic retrieval
- [x] LLM integration with grounded-answer instructions
- [x] Source references and similarity scores
- [x] Streamlit user interface
- [x] Setup guide and evaluation plan
