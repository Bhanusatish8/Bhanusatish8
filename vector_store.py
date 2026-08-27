"""
Vector store implementation using PostgreSQL + pgVector
"""
import os
from typing import List, Optional, Dict, Any
import openai
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import Document, QueryHistory, AgentLog, get_embedding_dimension
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Create a .env file with OPENAI_API_KEY=...")

openai.api_key = OPENAI_API_KEY


def embed_text(text: str) -> List[float]:
    """Generate embedding for text using OpenAI"""
    try:
        response = openai.Embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {str(e)}")


class PGVectorStore:
    """Vector store using PostgreSQL + pgVector"""

    def __init__(self, db: Session):
        self.db = db
        self.embedding_dim = get_embedding_dimension()

    def add_document(
        self,
        urn: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a document with embedding to the store"""
        try:
            # Generate embedding
            embedding = embed_text(text)

            # Create document
            doc = Document(
                urn=urn,
                text=text,
                embedding=embedding,
                metadata=metadata or {}
            )

            self.db.add(doc)
            self.db.commit()
            self.db.refresh(doc)

            return str(doc.id)
        except Exception as e:
            self.db.rollback()
            raise RuntimeError(f"Failed to add document: {str(e)}")

    def query_documents(
        self,
        query_text: str,
        top_k: int = 3,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Query documents by semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = embed_text(query_text)

            # Search using cosine similarity
            # pgVector uses <-> operator for cosine distance
            results = self.db.query(
                Document.id,
                Document.urn,
                Document.text,
                Document.metadata,
                # Calculate similarity (1 - distance for cosine)
                text(f"(1 - (embedding <-> CAST('{query_embedding}'::text AS vector))) as similarity")
            ).order_by(
                text(f"embedding <-> CAST('{query_embedding}'::text AS vector)")
            ).limit(top_k).all()

            # Format results
            formatted_results = []
            for doc_id, urn, doc_text, metadata, similarity in results:
                if similarity >= similarity_threshold:
                    formatted_results.append({
                        "doc_id": str(doc_id),
                        "urn": urn,
                        "text": doc_text,
                        "metadata": metadata,
                        "score": float(similarity)
                    })

            return formatted_results
        except Exception as e:
            raise RuntimeError(f"Failed to query documents: {str(e)}")

    def log_query(
        self,
        query_text: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """Log a query and its results"""
        try:
            query_embedding = embed_text(query_text)

            log = QueryHistory(
                query=query_text,
                embedding=query_embedding,
                answer=answer,
                retrieved_docs=retrieved_docs
            )

            self.db.add(log)
            self.db.commit()
            self.db.refresh(log)

            return str(log.id)
        except Exception as e:
            self.db.rollback()
            raise RuntimeError(f"Failed to log query: {str(e)}")

    def log_agent_request(
        self,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        status: str = "success",
        error_message: Optional[str] = None
    ) -> str:
        """Log agent request/response"""
        try:
            log = AgentLog(
                request_data=request_data,
                response_data=response_data,
                status=status,
                error_message=error_message
            )

            self.db.add(log)
            self.db.commit()
            self.db.refresh(log)

            return str(log.id)
        except Exception as e:
            self.db.rollback()
            raise RuntimeError(f"Failed to log agent request: {str(e)}")

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        try:
            doc = self.db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                return {
                    "id": str(doc.id),
                    "urn": doc.urn,
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at
                }
            return None
        except Exception as e:
            raise RuntimeError(f"Failed to get document: {str(e)}")

    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List all documents"""
        try:
            docs = self.db.query(Document).limit(limit).offset(offset).all()
            return [
                {
                    "id": str(doc.id),
                    "urn": doc.urn,
                    "text": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at
                }
                for doc in docs
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to list documents: {str(e)}")

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        try:
            self.db.query(Document).filter(Document.id == doc_id).delete()
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            raise RuntimeError(f"Failed to delete document: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        try:
            total_docs = self.db.query(Document).count()
            total_queries = self.db.query(QueryHistory).count()
            total_agent_logs = self.db.query(AgentLog).count()

            return {
                "total_documents": total_docs,
                "total_queries": total_queries,
                "total_agent_logs": total_agent_logs,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": self.embedding_dim
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get stats: {str(e)}")
