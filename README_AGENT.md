# Agentic AI (local)

This repository adds a minimal FastAPI service that:
- runs on localhost:17000
- uses your OpenAI API key from a `.env` file
- creates embeddings using OpenAI and stores them locally in `vector_store.json`
- exposes endpoints to ingest text/URNs, query with retrieval, and run a simple agent flow

Files added:
- `app.py` - the FastAPI application
- `requirements.txt` - Python requirements
- `.env.example` - example environment variables (copy to `.env` and add your key)

Quick start:
1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. (Optional) adjust EMBEDDING_MODEL and CHAT_MODEL in `.env`.
3. Install dependencies:
   pip install -r requirements.txt
4. Run the app:
   python app.py
5. Open http://localhost:17000/health to check service health.

Endpoints:
- POST /ingest  -> body: { "urn": "http://..." } or { "text": "..." }
- POST /query   -> body: { "query": "...", "top_k": 3 }
- POST /agent   -> body: { "urn": "http://...", "query": "..." }

Notes:
- This is intentionally minimal and for local/testing use. For production, add error handling, rate limiting, persistent DB, and secure secrets management.
