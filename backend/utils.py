# backend/utils.py
import numpy as np
from typing import List, Tuple
from sqlalchemy.orm import Session
from sklearn.manifold import TSNE
from models import Note
from embeddings import from_bytes

# Compute 2D coordinates for the Idea Map with small-dataset safeguards.

def tsne_2d(db: Session, perplexity: float = 10.0, random_state: int = 42) -> List[Tuple[int, float, float, str]]:
    """Return list of (id, x, y, title) for notes with embeddings.
    Robust to tiny datasets (n < perplexity)."""
    notes = db.query(Note).all()
    vecs, ids, titles = [], [], []
    for n in notes:
        if n.embedding: #only use notes with embeddings
            ids.append(n.id)
            titles.append(n.title or "")
            vecs.append(from_bytes(n.embedding.vector, n.embedding.dim))
    n = len(vecs)
    if n == 0:
        return []
    if n == 1:
        return [(ids[0], 0.0, 0.0, titles[0])]
    if n == 2:
        return [
            (ids[0], -1.0, 0.0, titles[0]),
            (ids[1],  1.0, 0.0, titles[1]),
        ]

    X = np.vstack(vecs)


    per = float(perplexity)
    per = min(per, max(2.0, (n - 1) / 3.0))  
    if per >= n:
        per = n - 1 - 1e-6  

    tsne = TSNE(
        n_components=2,
        perplexity=per,
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )
    Y = tsne.fit_transform(X)

    out: List[Tuple[int, float, float, str]] = []
    for i, (x, y) in enumerate(Y):
        out.append((ids[i], float(x), float(y), titles[i]))
    return out
