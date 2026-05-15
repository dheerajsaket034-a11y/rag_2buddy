"""
rag_engine.py
=============
Core RAG pipeline:
  1. PDF loading & text extraction
  2. Chunking with overlap
  3. Embedding via HuggingFace sentence-transformers (free, local)
  4. FAISS vector store (in-memory + optional disk persist)
  5. Retrieval + Gemini answer generation
"""

import os
import re
import hashlib
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 80 MB, runs on CPU, great quality
CHUNK_SIZE       = 500                    # characters per chunk
CHUNK_OVERLAP    = 80                     # overlap to preserve context
TOP_K            = 5                      # chunks to retrieve per query
VECTORSTORE_DIR  = Path("vectorstore")
VECTORSTORE_DIR.mkdir(exist_ok=True)

# Gemini safety / generation config
GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_output_tokens": 1024,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _file_hash(paths: List[str]) -> str:
    """Stable hash of a list of file paths (used for cache keys)."""
    h = hashlib.md5()
    for p in sorted(paths):
        h.update(Path(p).name.encode())
        h.update(str(Path(p).stat().st_size).encode())
    return h.hexdigest()[:12]


def extract_text_from_pdfs(pdf_paths: List[str]) -> List[Dict]:
    """
    Returns a list of page dicts:
      { 'source': filename, 'page': int, 'text': str }
    """
    pages = []
    for path in pdf_paths:
        reader = PdfReader(path)
        fname  = Path(path).name
        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            # Basic cleanup: collapse whitespace
            text = re.sub(r'\s+', ' ', raw).strip()
            if len(text) > 30:          # skip near-empty pages
                pages.append({"source": fname, "page": i, "text": text})
    return pages


def chunk_pages(pages: List[Dict],
                chunk_size: int = CHUNK_SIZE,
                overlap: int    = CHUNK_OVERLAP) -> List[Dict]:
    """
    Splits each page text into overlapping character-level chunks.
    Returns list of chunk dicts:
      { 'source', 'page', 'chunk_id', 'text' }
    """
    chunks = []
    cid = 0
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end  = min(start + chunk_size, len(text))
            snippet = text[start:end].strip()
            if len(snippet) > 40:
                chunks.append({
                    "source":   page["source"],
                    "page":     page["page"],
                    "chunk_id": cid,
                    "text":     snippet,
                })
                cid += 1
            start += chunk_size - overlap
    return chunks


# ── Embedding & FAISS ──────────────────────────────────────────────────────────

class RAGEngine:
    """Encapsulates the full RAG pipeline."""

    def __init__(self):
        self._embed_model: Optional[SentenceTransformer] = None
        self._index:  Optional[faiss.IndexFlatIP] = None   # Inner-product (cosine after norm)
        self._chunks: List[Dict] = []
        self._gemini: Optional[genai.GenerativeModel] = None
        self._loaded_hash: Optional[str] = None

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _get_embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        return self._embed_model

    def _get_gemini(self) -> genai.GenerativeModel:
        if self._gemini is None:
            api_key = os.getenv("GOOGLE_API_KEY", "")
            if not api_key or api_key == "your_google_gemini_api_key_here":
                raise ValueError(
                    "❌ GOOGLE_API_KEY not set. "
                    "Add it to your .env file or the Streamlit sidebar."
                )
            genai.configure(api_key=api_key)
            self._gemini = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=GENERATION_CONFIG,
            )
        return self._gemini

    def configure_api_key(self, api_key: str):
        """Allow runtime key injection from the Streamlit UI."""
        os.environ["GOOGLE_API_KEY"] = api_key
        self._gemini = None   # force rebuild with new key

    # ── Build index ───────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        model  = self._get_embed_model()
        vecs   = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return vecs.astype("float32")

    def _cache_path(self, fhash: str) -> Path:
        return VECTORSTORE_DIR / f"{fhash}.pkl"

    def build_index(self, pdf_paths: List[str],
                    chunk_size: int = CHUNK_SIZE,
                    overlap: int    = CHUNK_OVERLAP,
                    force_rebuild: bool = False) -> Tuple[int, int]:
        """
        Build (or load cached) FAISS index from a list of PDF paths.
        Returns (num_pages, num_chunks).
        """
        fhash     = _file_hash(pdf_paths)
        cache_pkl = self._cache_path(fhash)

        if not force_rebuild and cache_pkl.exists() and fhash == self._loaded_hash:
            return len(self._chunks), len(self._chunks)   # already loaded

        if not force_rebuild and cache_pkl.exists():
            with open(cache_pkl, "rb") as f:
                saved = pickle.load(f)
            self._chunks = saved["chunks"]
            self._index  = saved["index"]
            self._loaded_hash = fhash
            return saved["num_pages"], len(self._chunks)

        # Full rebuild
        pages  = extract_text_from_pdfs(pdf_paths)
        chunks = chunk_pages(pages, chunk_size, overlap)

        texts  = [c["text"] for c in chunks]
        vecs   = self._embed(texts)

        dim   = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)   # cosine similarity (vecs are L2-normed)
        index.add(vecs)

        self._chunks = chunks
        self._index  = index
        self._loaded_hash = fhash

        # Persist to disk
        with open(cache_pkl, "wb") as f:
            pickle.dump({"chunks": chunks, "index": index, "num_pages": len(pages)}, f)

        return len(pages), len(chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Returns top_k most relevant chunks with scores."""
        if self._index is None or not self._chunks:
            raise RuntimeError("Index not built. Please upload documents first.")

        q_vec = self._embed([query])
        scores, idxs = self._index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = dict(self._chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    # ── Generation ────────────────────────────────────────────────────────────

    def answer(self, query: str, top_k: int = TOP_K) -> Dict:
        """
        Full RAG answer:
          - retrieve relevant chunks
          - decide if query is in-scope
          - call Gemini for grounded answer
        Returns dict with keys: answer, chunks, in_scope
        """
        chunks = self.retrieve(query, top_k)

        # Relevance gate: if best chunk score < threshold, consider out-of-scope
        RELEVANCE_THRESHOLD = 0.30
        if not chunks or chunks[0]["score"] < RELEVANCE_THRESHOLD:
            return {
                "answer":   "⚠️ I couldn't find relevant information in your documents to answer this question. Please ask something covered in the uploaded material.",
                "chunks":   chunks,
                "in_scope": False,
            }

        # Build context block
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i} | {c['source']} | Page {c['page']}]\n{c['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are Study Buddy, a helpful and accurate academic assistant.
Your job is to answer questions STRICTLY based on the provided document excerpts below.

RULES:
1. Only use information present in the provided context. Do NOT use outside knowledge.
2. If the context does not contain enough information, say so clearly.
3. Cite source numbers like [Source 1], [Source 2] etc. in your answer.
4. Be concise, clear, and academically appropriate.
5. If the question is completely unrelated to the document content, politely refuse.

=== DOCUMENT CONTEXT ===
{context}

=== STUDENT QUESTION ===
{query}

=== YOUR ANSWER ==="""

        gemini = self._get_gemini()
        response = gemini.generate_content(prompt)
        answer_text = response.text.strip() if response.text else "No response generated."

        return {
            "answer":   answer_text,
            "chunks":   chunks,
            "in_scope": True,
        }

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    def stats(self) -> Dict:
        if not self.is_ready:
            return {}
        sources = list({c["source"] for c in self._chunks})
        return {
            "total_chunks": len(self._chunks),
            "sources":      sources,
            "num_docs":     len(sources),
        }
