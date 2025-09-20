# backend/models.py
from sqlalchemy import Column, Integer, String, Text, ForeignKey, LargeBinary 
from sqlalchemy.orm import relationship
from .db import Base  


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
