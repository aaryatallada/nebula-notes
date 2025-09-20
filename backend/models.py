# backend/models.py
from sqlalchemy import Column, Integer, String, Text, ForeignKey, LargeBinary 
from sqlalchemy.orm import relationship
from database import Base  




class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(String, index=True, nullable=True)  
    title = Column(String, nullable=True)
    content = Column(Text, nullable=True)

    embedding = relationship("Embedding", uselist=False, backref="note", cascade="all, delete-orphan")


class Embedding(Base):
    __tablename__ = "embeddings"
    note_id = Column(Integer, ForeignKey("notes.id"), primary_key=True)
    vector = Column(LargeBinary, nullable=False)  
    dim = Column(Integer, nullable=False)




# Model name is configurable via env; default is small & fast.
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Globals for lazy, thread-safe init
_MODEL = None
_MODEL_LOCK = threading.Lock()

# We expose the dimension so other modules can reference it if needed
EMBED_DIM = 384  # MiniLM-L6-v2 -> 384; will be verified at load


def _load_model() -> SentenceTransformer:
    """Create and return a SentenceTransformer model."""
    model = SentenceTransformer(MODEL_NAME)
    # Probe one vector to confirm dim and adjust EMBED_DIM if needed
    vec = model.encode(["_probe_"], convert_to_numpy=True, normalize_embeddings=True)
    global EMBED_DIM
    EMBED_DIM = int(vec.shape[-1])
    return model


def get_model() -> SentenceTransformer:
    """Thread-safe singleton model getter."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = _load_model()
    return _MODEL


def embed_texts(texts, batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    """
    Embed a list of strings -> (N, D) float32 numpy array.
    - normalize=True yields unit vectors so cosine == dot product.
    """
    model = get_model()
    arr = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return arr.astype(np.float32)