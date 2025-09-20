# backend/app.py

from typing import List, Optional 

from fastapi import FastAPI, Depends, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Note
from schemas import NoteCreate, NoteUpdate, NoteOut, SearchResult, MapPoint
from utils import tsne_2d


from .embeddings import ensure_embedding, rebuild_embeddings, similarity_search
try:
    from embeddings import hybrid_search, rerank 
    _HAS_HYBRID = True
except Exception:
    hybrid_search = None  
    rerank = None         
    _HAS_HYBRID = False



def get_owner_id(x_user: str | None = Header(default=None, alias="X-User")) -> str:
    if not x_user:
        raise HTTPException(400, "Missing X-User header")
    return x_user


def _normalize_scores(pairs):
    """Min-max scale scores to [0,1] for display."""
    if not pairs:
        return []
    vals = [s for _, s in pairs]
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        norm = [0.5 for _ in vals]
    else:
        norm = [(v - mn) / (mx - mn) for v in vals]
    return [(pairs[i][0], float(norm[i])) for i in range(len(pairs))]



Base.metadata.create_all(bind=engine)

app = FastAPI(title="Nebula Notes", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_all_embeddings(db: Session, owner: str | None = None) -> int:
    q = db.query(Note)
    if owner is not None:
        q = q.filter(Note.owner_id == owner)
    notes = q.all()
    for n in notes:

        ensure_embedding(db, n)
    db.commit()
    return len(notes)


@app.get("/api/health")
def health():
    return {"ok": True, "hybrid_search": _HAS_HYBRID}


@app.get("/api/notes", response_model=List[NoteOut])
def list_notes(owner: str = Depends(get_owner_id), db: Session = Depends(get_db)):
    return (db.query(Note)
              .filter(Note.owner_id == owner)
              .order_by(Note.id.desc())
              .all())


@app.post("/api/notes")
def create_note(request: Request, payload: dict):
    owner = get_owner_id(request)  # your helper that reads X-User
    title = (payload.get("title") or "").strip()
    content = (payload.get("content") or "").strip()

    with SessionLocal() as db:
        note = Note(owner_id=owner, title=title, content=content)
        db.add(note)
        db.commit()
        db.refresh(note)

        # NEW: upsert the embedding for this note
        ensure_embedding(db, note)
        db.commit()

        return note


@app.get("/api/notes/{note_id}", response_model=NoteOut)
def get_note(note_id: int, db: Session = Depends(get_db)):
    """Fetch a single note by id."""
    n = db.get(Note, note_id)
    if not n:
        raise HTTPException(404, "Note not found")
    return n


@app.put("/api/notes/{note_id}", response_model=NoteOut)
def update_note(note_id: int, payload: NoteUpdate, db: Session = Depends(get_db)):
    """Update a note and refresh its embedding."""
    n = db.get(Note, note_id)
    if not n:
        raise HTTPException(404, "Note not found")
    if payload.title is not None:
        n.title = payload.title
    if payload.content is not None:
        n.content = payload.content
    db.add(n)
    db.commit()
    db.refresh(n)
    ensure_embedding(db, n)  
    db.commit()
    db.refresh(n)
    return n


@app.post("/api/embed/rebuild")
def rebuild_embeddings_endpoint(request: Request):
    owner = get_owner_id(request)  # X-User header
    with SessionLocal() as db:
        rebuild_embeddings(db, owner=owner)
    return {"status": "ok"}


@app.delete("/api/notes/{note_id}")
def delete_note(note_id: int, db: Session = Depends(get_db)):
    """Delete a note (and its embedding via cascade)."""
    n = db.get(Note, note_id)
    if not n:
        raise HTTPException(404, "Note not found")
    db.delete(n)
    db.commit()
    return {"deleted": note_id}


@app.post("/api/embed/rebuild")
def rebuild_embeddings(db: Session = Depends(get_db)):
    """
    Recompute embeddings for all notes.
    Useful after bulk edits or first-time model download.
    """
    count = _ensure_all_embeddings(db)
    return {"count": count, "status": "rebuilt"}



@app.get("/api/search", response_model=List[SearchResult])
def search(q: str, k: int = 10, mode: str = Query("hybrid", description="hybrid|semantic"),
           alpha: Optional[float] = Query(None, ge=0.0, le=1.0),
           owner: str = Depends(get_owner_id), db: Session = Depends(get_db)):
    _ensure_all_embeddings(db, owner)

    if mode == "semantic" or not _HAS_HYBRID:
        ranked = similarity_search(db, q, k=k, owner = owner)
    else:
        ranked = hybrid_search(db, q, k=k, alpha=alpha, owner = owner) 
        if rerank is not None:
            ranked = rerank(q, ranked)  

    ranked = _normalize_scores(ranked)

    return [{"note": n, "score": float(score)} for (n, score) in ranked]


@app.get("/api/map", response_model=List[MapPoint])
def idea_map(owner: str = Depends(get_owner_id), db: Session = Depends(get_db)):
    _ensure_all_embeddings(db, owner)
    coords = tsne_2d(db, owner=owner)
    return [{"id": i, "x": x, "y": y, "title": t} for (i, x, y, t) in coords]
