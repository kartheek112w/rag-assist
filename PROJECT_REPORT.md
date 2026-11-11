# Project Report: Chat With Your Documents

## 1. Abstract
Chat With Your Documents is a Retrieval-Augmented Generation (RAG) application that allows users to ask natural-language questions about their own files. It extracts text from PDF, DOCX, TXT, and Markdown documents, divides the content into overlapping chunks, converts each chunk into an embedding, and stores the vectors in a FAISS index. When a user asks a question, the system embeds the query, retrieves the most semantically similar chunks, and provides those passages to an LLM. The model generates an answer grounded in the supplied context and includes source references. This design helps reduce hallucinations by giving the model relevant evidence and instructing it to acknowledge when the answer is not available. The application includes a Streamlit interface, persistent local indexing, configurable retrieval count, and visible source passages.

## 2. Problem statement
General-purpose LLMs may produce confident but unsupported answers. Users also need a convenient way to search large documents without manually locating relevant paragraphs. This project addresses both needs by retrieving evidence from user-provided documents before generating an answer.

## 3. Objectives
- Understand LLM hallucinations and RAG.
- Learn embeddings, semantic similarity, chunking, and vector search.
- Process common document formats.
- Build a document-based question-answering workflow.
- Display evidence so users can verify generated answers.

## 4. Technology stack
| Component | Technology |
|---|---|
| User interface | Streamlit |
| Language | Python |
| Document extraction | pypdf, python-docx |
| Embeddings | Sentence Transformers — all-MiniLM-L6-v2 |
| Vector search | FAISS |
| LLM | OpenAI Chat Completions API |
| Configuration | python-dotenv |

## 5. System workflow
1. Document upload: user uploads supported files.
2. Extraction: text is extracted from each file.
3. Cleaning: whitespace and unwanted null characters are normalized.
4. Chunking: text is split into 900-word chunks with 150-word overlap.
5. Embedding: each chunk is transformed into a dense vector.
6. Indexing: vectors and metadata are saved in FAISS and JSON.
7. Query embedding: the user's question is embedded with the same model.
8. Retrieval: nearest chunks are selected using similarity search.
9. Generation: retrieved context is sent to the LLM with grounding instructions.
10. Answer: the response and source passages are shown in the interface.

## 6. Hallucination reduction
The system reduces hallucination risk by restricting the answer to retrieved context, using a low temperature, asking the model to state when evidence is missing, and showing source passages. These controls do not eliminate hallucinations; users should verify important claims against the original documents.

## 7. Retrieval quality improvements
Chunk size, overlap, and top-k should be evaluated using representative questions. Hybrid search, metadata filtering, reranking, OCR for scanned files, and domain-specific embedding models are possible future enhancements.

## 8. Testing plan
- Upload a text file and confirm it is indexed.
- Upload a PDF and DOCX and check extraction.
- Ask a question whose answer appears explicitly in a document.
- Ask a question that requires matching similar wording.
- Ask a question absent from the documents and verify the app abstains.
- Inspect the source chunks and similarity scores.

## 9. Expected outcome
A working local application that answers questions about uploaded documents using retrieved evidence, while exposing source passages for verification.

## 10. Conclusion
The project demonstrates the complete RAG pipeline from document ingestion to grounded LLM response. It provides a practical foundation for knowledge assistants, policy search, study-material Q&A, and internal document support systems.
