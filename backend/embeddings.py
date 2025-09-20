# backend/embeddings.py
#embedding storage + search (semantic, hybrid) + optional cross-encoder reranker.

from typing import List, Tuple, Optional
import re
import numpy as np
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re

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
    return vec.astype(np.float32).tobytes()

def from_bytes(b: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32, count=dim)

def _upsert_embedding(db: Session, note_id: int, vector: np.ndarray):
    dim = int(vector.shape[-1])
    existing = db.get(Embedding, note_id)  # SQLAlchemy 2.x way
    if existing:
        existing.vector = to_bytes(vector)
        existing.dim = dim
        # no add(); it’s already in the session
    else:
        db.add(Embedding(note_id=note_id, vector=to_bytes(vector), dim=dim))

def ensure_embedding(db: Session, note: Note):
    """
    Compute/update a single note's vector and upsert into DB.
    Assumes you have embed_texts([...]) available in this module.
    """
    text = f"{note.title or ''}\n{note.content or ''}".strip()
    vec = embed_texts([text])[0]  # your existing embedding function
    _upsert_embedding(db, note.id, vec)

def rebuild_embeddings(db: Session, owner: Optional[str] = None):
    """
    Re-embed all notes for this owner (or all notes if owner is None)
    and upsert rows to avoid PK collisions.
    """
    q = db.query(Note)
    if owner is not None:
        q = q.filter(Note.owner_id == owner)
    notes = q.all()
    if not notes:
        return

    texts = [f"{n.title or ''}\n{n.content or ''}".strip() for n in notes]
    vecs = embed_texts(texts)  # same embed_texts you already use

    for n, v in zip(notes, vecs):
        _upsert_embedding(db, n.id, v)

    db.commit()

# Embedding utilities
def embed_texts(texts: List[str]) -> np.ndarray:
    """Batch-embed a list of strings -> (n, d) array (L2-normalized)."""
    model = get_sentence_model()
    return np.array(model.encode(texts, normalize_embeddings=True))



def missing_embeddings(db: Session) -> List[Note]:
    """Return notes that currently have no stored embedding."""
    notes = db.query(Note).all()
    return [n for n in notes if not n.embedding]

def similarity_search(
    db: Session,
    query: str,
    k: int = 10,
    owner: Optional[str] = None,
) -> List[Tuple[Note, float]]:
    """
    Pure semantic search scoped to an owner (if provided).
    """
    q = db.query(Note)
    if owner is not None:
        q = q.filter(Note.owner_id == owner)
    notes = q.all()

    meta: List[Note] = []
    vecs: List[np.ndarray] = []
    for n in notes:
        if n.embedding:
            meta.append(n)
            vecs.append(from_bytes(n.embedding.vector, n.embedding.dim))
    if not vecs:
        return []

    V = np.vstack(vecs)
    q_vec = embed_texts([query])[0].reshape(1, -1)
    sims = cosine_similarity(q_vec, V)[0]
    ranked = sorted(zip(meta, sims), key=lambda t: t[1], reverse=True)
    return ranked[:k]

def _tok(s: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", (s or "").lower()) if t]
def hybrid_search(
    db: Session,
    query: str,
    k: int = 10,
    alpha: Optional[float] = None,
    owner: Optional[str] = None,
) -> List[Tuple[Note, float]]:
    """
    Hybrid retrieval (semantic + BM25 + priors + PRF), scoped to an owner (if provided).
    """
    # 1) Fetch only this owner's notes
    q_notes = db.query(Note)
    if owner is not None:
        q_notes = q_notes.filter(Note.owner_id == owner)
    notes = q_notes.all()

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

    V = np.vstack(vecs)

    # 2) Semantic
    q_vec = embed_texts([query])[0].reshape(1, -1)
    sem = cosine_similarity(q_vec, V)[0]

    # 3) BM25 corpus with title boost
    corpus_tokens: List[List[str]] = []
    doc_lengths: List[int] = []
    for t, c in zip(titles, contents):
        t_tok = _tok(t)
        c_tok = _tok(c)
        toks = t_tok + t_tok + c_tok
        corpus_tokens.append(toks)
        doc_lengths.append(len(toks))

    bm25 = BM25Okapi(corpus_tokens)
    q_tokens = _tok(query)

    # 4) PRF for zero-overlap short queries
    STOP = {
        "the","a","an","of","and","or","to","for","in","on","at","is","it",
        "this","that","with","by","as","be","are","was","were","from","up",
        "new","note","notes"
    }
    def prf_terms(top_m: int = 3, topk_terms: int = 6) -> List[str]:
        idx = np.argsort(-sem)[:top_m]
        counts = {}
        for i in idx:
            for tok in corpus_tokens[i]:
                if len(tok) < 3 or tok in STOP:
                    continue
                counts[tok] = counts.get(tok, 0) + 1
        return [t for t, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:topk_terms]]

    query_tokens_for_bm25 = list(q_tokens)
    if len(q_tokens) <= 2:
        qset = set(q_tokens)
        has_overlap_any = any(bool(qset.intersection(doc)) for doc in corpus_tokens)
        if not has_overlap_any:
            exp = prf_terms(top_m=3, topk_terms=6)
            query_tokens_for_bm25 += exp + exp

    bm25_scores = bm25.get_scores(query_tokens_for_bm25).astype(float)
    bmax = float(bm25_scores.max()) if bm25_scores.size else 0.0
    bm25_norm = (bm25_scores / bmax) if bmax > 0 else np.zeros_like(bm25_scores)

    # 5) Priors and penalties
    L = np.array(doc_lengths, dtype=float)
    length_prior = np.clip((L - 8) / 60.0, 0.0, 1.0)
    beta = 0.15

    qset = set(query_tokens_for_bm25 if len(q_tokens) <= 2 else q_tokens)
    overlap = np.array([1.0 if qset.intersection(doc) else 0.0 for doc in corpus_tokens], dtype=float)
    gamma = 0.18 if len(q_tokens) <= 2 else 0.0

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
