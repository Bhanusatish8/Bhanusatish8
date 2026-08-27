"""
FastAPI application with PostgreSQL + pgVector integration
"""
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import openai
from sqlalchemy.orm import Session

from database import init_db, get_db
from vector_store import PGVectorStore, embed_text

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
PGVECTOR_ENABLED = os.getenv("PGVECTOR_ENABLED", "true").lower() == "true"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Create a .env file with OPENAI_API_KEY=...")

openai.api_key = OPENAI_API_KEY

# Initialize FastAPI app
app = FastAPI(
    title="Agentic AI with pgVector",
    description="RAG system with OpenAI embeddings and PostgreSQL + pgVector backend",
    version="1.0.0"
)


# Initialize database
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    if PGVECTOR_ENABLED:
        init_db()
        print("✓ Database initialized")
    print("✓ Application started successfully")


# Pydantic models
class IngestRequest(BaseModel):
    urn: str
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    similarity_threshold: float = 0.0


class AgentRequest(BaseModel):
    urn: Optional[str] = None
    query: Optional[str] = None


class DocumentResponse(BaseModel):
    doc_id: str
    urn: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    retrieved: List[DocumentResponse]


# Helper functions
def chat_with_context(prompt: str) -> str:
    """Generate response using OpenAI with context"""
    try:
        response = openai.ChatCompletion.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that uses provided context to answer user questions concisely and accurately."
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


# API Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "pgvector_enabled": PGVECTOR_ENABLED,
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "chat_model": CHAT_MODEL
    }


@app.post("/ingest", response_model=Dict[str, Any])
async def ingest(req: IngestRequest, db: Session = Depends(get_db)):
    """Ingest a document with embeddings"""
    try:
        if not req.text:
            raise HTTPException(status_code=400, detail="text is required")

        store = PGVectorStore(db)
        doc_id = store.add_document(
            urn=req.urn,
            text=req.text,
            metadata=req.metadata
        )

        return {
            "status": "ingested",
            "doc_id": doc_id,
            "urn": req.urn
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest error: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, db: Session = Depends(get_db)):
    """Query documents using semantic search"""
    try:
        if not req.query:
            raise HTTPException(status_code=400, detail="query is required")

        store = PGVectorStore(db)

        # Search for similar documents
        results = store.query_documents(
            query_text=req.query,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold
        )

        # Build context from retrieved documents
        context_parts = []
        retrieved = []

        for result in results:
            context_parts.append(
                f"URN: {result['urn']}\nScore: {result['score']:.4f}\nText:\n{result['text']}\n---\n"
            )
            retrieved.append(
                DocumentResponse(
                    doc_id=result['doc_id'],
                    urn=result['urn'],
                    score=result['score']
                )
            )

        # Generate answer using context
        prompt = f"""Use the following retrieved contexts to answer the user's question. If the context is insufficient, say you don't have enough information.

CONTEXTS:

{chr(10).join(context_parts)}

User question:
{req.query}
"""

        answer = chat_with_context(prompt)

        # Log the query
        try:
            store.log_query(req.query, answer, [r.dict() for r in retrieved])
        except Exception as e:
            print(f"Warning: Failed to log query: {e}")

        return QueryResponse(answer=answer, retrieved=retrieved)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.post("/agent", response_model=Dict[str, Any])
async def agent(req: AgentRequest, db: Session = Depends(get_db)):
    """Agent endpoint - ingest and/or query"""
    try:
        store = PGVectorStore(db)
        ingest_result = None

        # Ingest if URN provided
        if req.urn:
            # For demo purposes, create dummy text from URN
            # In production, you'd fetch from the URN
            text = f"Content from {req.urn}"
            doc_id = store.add_document(
                urn=req.urn,
                text=text,
                metadata={"source": "uri"}
            )
            ingest_result = {"doc_id": doc_id, "urn": req.urn}

            if not req.query:
                log_id = store.log_agent_request(
                    request_data=req.dict(),
                    response_data={"ingest": ingest_result},
                    status="success"
                )
                return {
                    "status": "ingested",
                    "ingest": ingest_result,
                    "log_id": log_id
                }

        # Query if query provided
        if not req.query:
            raise HTTPException(status_code=400, detail="query is required when not just ingesting")

        # Execute query
        query_req = QueryRequest(query=req.query, top_k=3)
        results = store.query_documents(
            query_text=query_req.query,
            top_k=query_req.top_k
        )

        context_parts = []
        for result in results:
            context_parts.append(
                f"URN: {result['urn']}\nScore: {result['score']:.4f}\nText:\n{result['text']}\n---\n"
            )

        prompt = f"""Use the following retrieved contexts to answer the user's question.

CONTEXTS:

{chr(10).join(context_parts)}

User question:
{req.query}
"""

        answer = chat_with_context(prompt)

        response_data = {
            "answer": answer,
            "retrieved": [{"doc_id": r['doc_id'], "urn": r['urn'], "score": r['score']} for r in results]
        }

        # Log agent request
        log_id = store.log_agent_request(
            request_data=req.dict(),
            response_data=response_data,
            status="success"
        )

        return {
            "status": "ok",
            "result": response_data,
            "ingest": ingest_result,
            "log_id": log_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/documents", response_model=Dict[str, Any])
async def list_documents(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    """List all ingested documents"""
    try:
        store = PGVectorStore(db)
        documents = store.list_documents(limit=limit, offset=offset)
        return {
            "documents": documents,
            "total": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        store = PGVectorStore(db)
        return store.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.delete("/documents/{doc_id}", response_model=Dict[str, Any])
async def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """Delete a document"""
    try:
        store = PGVectorStore(db)
        success = store.delete_document(doc_id)
        return {"status": "deleted" if success else "not_found", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=17000)
