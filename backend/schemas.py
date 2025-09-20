from pydantic import BaseModel
from typing import Optional, List

# Define the shapes of data used in API requests and responses

class NoteCreate(BaseModel):
    title: str
    content: str

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None

class NoteOut(BaseModel):
    id: int
    owner_id: str
    title: str
    content: str
    class Config:
        from_attributes = True

class SearchResult(BaseModel):
    note: NoteOut
    score: float

class MapPoint(BaseModel):
    id: int
    x: float
    y: float
    title: str
