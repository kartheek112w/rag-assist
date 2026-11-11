import io
import json
import re
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document as DocxDocument

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "vector_store"
INDEX_PATH = STORE_DIR / "documents.faiss"
CHUNKS_PATH = STORE_DIR / "chunks.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        doc = DocxDocument(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {suffix}")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Split text into word-based chunks with a controlled overlap."""
    words = clean_text(text).split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


class DocumentRAG:
    def __init__(self):
        STORE_DIR.mkdir(exist_ok=True)
        self.model = SentenceTransformer(MODEL_NAME)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self._load()

    def _load(self):
        if INDEX_PATH.exists() and CHUNKS_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            self.chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

    def _save(self):
        faiss.write_index(self.index, str(INDEX_PATH))
        CHUNKS_PATH.write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_documents(self, documents: List[Tuple[str, bytes]]) -> int:
        new_chunks = []
        for filename, content in documents:
            text = extract_text(filename, content)
            pieces = chunk_text(text)
            for number, piece in enumerate(pieces, start=1):
                new_chunks.append({
                    "file": filename,
                    "chunk": number,
                    "text": piece,
                })
        if not new_chunks:
            raise ValueError("No readable text was found in the uploaded files.")
        vectors = self.model.encode(
            [item["text"] for item in new_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        self.index.add(vectors)
        self.chunks.extend(new_chunks)
        self._save()
        return len(new_chunks)

    def retrieve(self, query: str, top_k: int = 4):
        if self.index.ntotal == 0:
            return []
        vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, ids = self.index.search(vector, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                item = dict(self.chunks[int(idx)])
                item["score"] = float(score)
                results.append(item)
        return results

    def stats(self):
        return {
            "chunks": len(self.chunks),
            "documents": len({item["file"] for item in self.chunks}),
        }
