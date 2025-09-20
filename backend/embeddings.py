# backend/embeddings.py
#embedding storage + search (semantic, hybrid) + optional cross-encoder reranker.

from typing import List, Tuple, Optional
import re
import numpy as np
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from models import Note, Embedding


# Models onlly have to downloaded once, then cached locally
SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_sentence_model: Optional[SentenceTransformer] = None
_rerank_model: Optional[CrossEncoder] = None

def get_sentence_model() -> SentenceTransformer:
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)
    return _sentence_model

def get_rerank_model() -> CrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = CrossEncoder(RERANK_MODEL_NAME)
    return _rerank_model

# Serialization helpers
def to_bytes(vec: np.ndarray) -> bytes:
    """Serialize float32 vector to bytes for SQLite BLOB (BytesIO-safe)."""
    return vec.astype(np.float32).tobytes()

def from_bytes(b: bytes, dim: int) -> np.ndarray:
    """Deserialize bytes to float32 numpy vector of expected dim."""
    arr = np.frombuffer(b, dtype=np.float32, count=dim)
    if arr.size != dim:
        raise ValueError(f"Embedding size mismatch: expected {dim}, got {arr.size}")
    return arr

# Embedding utilities
def embed_texts(texts: List[str]) -> np.ndarray:
    """Batch-embed a list of strings -> (n, d) array (L2-normalized)."""
    model = get_sentence_model()
    return np.array(model.encode(texts, normalize_embeddings=True))

def ensure_embedding(db: Session, note: Note) -> None:
    """
    Create or update embedding for a single note.
    Idempotent: will overwrite existing embedding if present.
    """
    text = f"{note.title or ''}\n{note.content or ''}"
    vec = embed_texts([text])[0]
    dim = int(vec.shape[0])
    blob = to_bytes(vec)
    if note.embedding:
        note.embedding.vector = blob
        note.embedding.dim = dim
    else:
        db.add(Embedding(note_id=note.id, vector=blob, dim=dim))

def missing_embeddings(db: Session) -> List[Note]:
    """Return notes that currently have no stored embedding."""
    notes = db.query(Note).all()
    return [n for n in notes if not n.embedding]

def similarity_search(db: Session, query: str, k: int = 10) -> List[Tuple[Note, float]]:
    """Return top-k notes by cosine similarity between query and note embeddings."""
    notes = db.query(Note).all()
    meta: List[Note] = []
    vecs: List[np.ndarray] = []
    for n in notes:
        if n.embedding:
            meta.append(n)
            vecs.append(from_bytes(n.embedding.vector, n.embedding.dim))
    if not vecs:
        return []
    M = np.vstack(vecs)  # (N, d)
    q_vec = embed_texts([query])[0].reshape(1, -1)
    sims = cosine_similarity(q_vec, M)[0]  # (N,)
    ranked = sorted(zip(meta, sims), key=lambda t: t[1], reverse=True)
    return ranked[:k]

def _tok(s: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", (s or "").lower()) if t]

def hybrid_search(db: Session, query: str, k: int = 10, alpha: Optional[float] = None) -> List[Tuple[Note, float]]:
    """
    Hybrid retrieval with pragmatic tweaks + pseudo-relevance feedback (PRF):

      final = α * semantic + (1-α) * bm25_norm + β * length_prior - γ * no_overlap_penalty

    - Dynamic α: short queries (<= 2 tokens) rely more on BM25.
    - Title boost: title tokens duplicated in corpus.
    - Length prior: down-weight very short notes.
    - No-overlap penalty: for very short queries, docs with zero lexical overlap drop.
    - PRF: if there is zero lexical overlap, expand query tokens using top semantic docs.
    """
    notes = db.query(Note).all()
    meta: List[Note] = []
    vecs: List[np.ndarray] = []
    titles: List[str] = []
    contents: List[str] = []

    for n in notes:
        if n.embedding:
            meta.append(n)
            vecs.append(from_bytes(n.embedding.vector, n.embedding.dim))
            titles.append(n.title or "")
            contents.append(n.content or "")

    if not vecs:
        return []

    V = np.vstack(vecs)  # (N, d)

    # Semantic part
    q_vec = embed_texts([query])[0].reshape(1, -1)
    sem = cosine_similarity(q_vec, V)[0]  

    corpus_tokens: List[List[str]] = []
    doc_lengths: List[int] = []
    for t, c in zip(titles, contents):
        t_tok = _tok(t)
        c_tok = _tok(c)
        tokens = t_tok + t_tok + c_tok  # boost
        corpus_tokens.append(tokens)
        doc_lengths.append(len(tokens))

    bm25 = BM25Okapi(corpus_tokens)
    q_tokens = _tok(query)

    STOP = {
        "the","a","an","of","and","or","to","for","in","on","at","is","it",
        "this","that","with","by","as","be","are","was","were","from","up",
        "new","note","notes"
    }
    def prf_terms(top_m: int = 3, topk_terms: int = 6) -> List[str]:
        # pick tokens from top-m semantically similar docs
        idx = np.argsort(-sem)[:top_m]
        counts = {}
        for i in idx:
            for tok in corpus_tokens[i]:
                if len(tok) < 3 or tok in STOP:
                    continue
                counts[tok] = counts.get(tok, 0) + 1
        # top-k frequent terms
        return [t for t, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:topk_terms]]

    query_tokens_for_bm25 = list(q_tokens)

    if len(q_tokens) <= 2:
        qset = set(q_tokens)
        has_overlap_any = any(bool(qset.intersection(doc)) for doc in corpus_tokens)
        if not has_overlap_any:
            exp = prf_terms(top_m=3, topk_terms=6)
            query_tokens_for_bm25 += exp + exp  # duplicate to give them weight

    # BM25 scores (normalized to [0,1])
    bm25_scores = bm25.get_scores(query_tokens_for_bm25).astype(float)
    bmax = float(bm25_scores.max()) if bm25_scores.size else 0.0
    bm25_norm = (bm25_scores / bmax) if bmax > 0 else np.zeros_like(bm25_scores)

    L = np.array(doc_lengths, dtype=float)
    length_prior = np.clip((L - 8) / 60.0, 0.0, 1.0)
    beta = 0.15

    qset = set(query_tokens_for_bm25 if len(q_tokens) <= 2 else q_tokens)
    overlap = np.array([1.0 if qset.intersection(doc) else 0.0 for doc in corpus_tokens], dtype=float)
    gamma = 0.18 if len(q_tokens) <= 2 else 0.0

    # Dynamic alpha
    if alpha is None:
        alpha = 0.35 if len(q_tokens) <= 2 else 0.65

    combined = alpha * sem + (1.0 - alpha) * bm25_norm + beta * length_prior - gamma * (1.0 - overlap)

    ranked = sorted(zip(meta, combined), key=lambda t: t[1], reverse=True)
    return ranked[:k]


#Cross-encoder reranker 
def rerank(query: str, pairs: List[Tuple[Note, float]]) -> List[Tuple[Note, float]]:
    """
    Re-score (top-N) with a cross-encoder. Input is a list of (Note, score).
    Returns a list of (Note, rerank_score) sorted desc.
    """
    if not pairs:
        return []
    model = get_rerank_model()
    texts = [(query, f"{n.title}\n{n.content}") for (n, _) in pairs]
    scores = model.predict(texts)  # higher is better
    out = [(pairs[i][0], float(scores[i])) for i in range(len(pairs))]
    out.sort(key=lambda t: t[1], reverse=True)
    return out
