"""依賴注入"""

from src.db.database import SessionLocal


def get_db():
    """FastAPI dependency: DB session"""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
