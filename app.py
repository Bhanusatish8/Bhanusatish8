from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import uuid
import requests
from dotenv import load_dotenv
import openai
import numpy as np

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Create a .env file with OPENAI_API_KEY=...")

openai.api_key = OPENAI_API_KEY

app = FastAPI(title="Agentic AI (local)")

STORE_FILE = "vector_store.json"


def _cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class VectorStore:
    def __init__(self, path=STORE_FILE):
        self.path = path
        self._data = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except FileNotFoundError:
            self._data = {}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def add(self, urn: str, text: str, metadata: dict | None = None):
        doc_id = str(uuid.uuid4())
        embedding = embed_text(text)
        self._data[doc_id] = {
            "urn": urn,
            "text": text,
            "metadata": metadata or {},
            "embedding": embedding,
        }
        self._save()
        return doc_id

    def query(self, embedding, top_k=3):
        scores = []
        for doc_id, doc in self._data.items():
            score = _cosine_similarity(embedding, doc["embedding"])
            scores.append((score, doc_id, doc))
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[:top_k]


store = VectorStore()


class IngestRequest(BaseModel):
    urn: str | None = None
    text: str | None = None
    metadata: dict | None = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class AgentRequest(BaseModel):
    urn: str | None = None
    query: str | None = None


def embed_text(text: str):
    # OpenAI embeddings
    resp = openai.Embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp["data"][0]["embedding"]


def chat_with_context(prompt: str):
    resp = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an assistant that uses provided context to answer user questions concisely."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=800,
    )
    return resp["choices"][0]["message"]["content"].strip()


@app.post("/ingest")
async def ingest(req: IngestRequest):
    if not req.text and not req.urn:
        raise HTTPException(status_code=400, detail="Provide either text or urn to ingest")

    text = req.text
    urn = req.urn or "local_text"
    if req.urn and not req.text:
        # Fetch from URN (support http/https)
        try:
            r = requests.get(req.urn, timeout=10)
            r.raise_for_status()
            text = r.text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URN: {e}")

    doc_id = store.add(urn=urn, text=text, metadata=req.metadata)
    return {"status": "ingested", "doc_id": doc_id}


@app.post("/query")
async def query(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="query is required")

    q_emb = embed_text(req.query)
    top = store.query(q_emb, top_k=req.top_k)

    context_parts = []
    for score, doc_id, doc in top:
        context_parts.append(f"URN: {doc['urn']}\nScore: {score:.4f}\nText:\n{doc['text']}\n---\n")

    prompt = """
Use the following retrieved contexts to answer the user's question. If the context is insufficient, say you don't have enough information.

CONTEXTS:

""" + "\n".join(context_parts) + f"\nUser question:\n{req.query}\n"

    answer = chat_with_context(prompt)
    return {"answer": answer, "retrieved": [ {"doc_id": d[1], "score": d[0], "urn": d[2]["urn"]} for d in top ]}


@app.post("/agent")
async def agent(req: AgentRequest):
    # Simple agent: ingest URN if provided, then run a query using provided query or return ingest result
    if req.urn:
        # ingest the URN content
        ingest_req = IngestRequest(urn=req.urn)
        ingest_resp = await ingest(ingest_req)
        if not req.query:
            return {"status": "ingested", "ingest": ingest_resp}

    if not req.query:
        raise HTTPException(status_code=400, detail="query is required when not just ingesting")

    qreq = QueryRequest(query=req.query, top_k=3)
    qresp = await query(qreq)
    return {"status": "ok", "result": qresp}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=17000)
