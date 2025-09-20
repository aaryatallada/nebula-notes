from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.sqlite import BLOB

from database import Base

class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(String(64), index=True, nullable=False)  
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    # One-to-one relationship with Embedding so deleting a Note also deletes its Embedding
    embedding = relationship("Embedding", back_populates="note", uselist=False, cascade="all, delete-orphan")

class Embedding(Base):
    __tablename__ = "embeddings"

    note_id = Column(Integer, ForeignKey("notes.id"), primary_key=True)
    # store raw bytes for the float32 vector
    vector = Column(BLOB, nullable=False)
    dim = Column(Integer, nullable=False)

    note = relationship("Note", back_populates="embedding")
