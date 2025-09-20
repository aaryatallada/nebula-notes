from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Create the db
DATABASE_URL = "sqlite:///./nebula.db"

# Create engine, since Uvicorn may spawn multiple workers/threads ensure check_same_thread is False
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
# DB session for each request
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base for model classes
Base = declarative_base()

def get_db():
    """FastAPI dependency that yields a DB session and ensures it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
