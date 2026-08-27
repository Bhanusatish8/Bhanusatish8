"""
Database configuration and utilities for PostgreSQL + pgVector
"""
import os
from typing import Optional
from sqlalchemy import create_engine, Column, String, Text, DateTime, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/agentic_ai_db")
PGVECTOR_ENABLED = os.getenv("PGVECTOR_ENABLED", "true").lower() == "true"

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


class Document(Base):
    """Document model with vector embeddings"""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    urn = Column(String(500), nullable=False, index=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    metadata = Column(JSONB, nullable=True, default={})
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class QueryHistory(Base):
    """Query history model"""
    __tablename__ = "query_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    answer = Column(Text, nullable=True)
    retrieved_docs = Column(JSONB, nullable=True, default=[])
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow, index=True)


class AgentLog(Base):
    """Agent logs model"""
    __tablename__ = "agent_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_data = Column(JSONB, nullable=True)
    response_data = Column(JSONB, nullable=True)
    status = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow, index=True)


def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialized successfully")


def get_embedding_dimension() -> int:
    """Get embedding dimension (OpenAI uses 1536 for text-embedding-3-small)"""
    return 1536
