# Agentic AI with pgVector - Complete Setup Guide

This guide will help you set up and run the project with Poetry, PostgreSQL, and pgVector for embeddings.

## Prerequisites

- Python 3.10 or higher
- Docker & Docker Compose (for PostgreSQL + pgVector)
- Poetry (Python dependency manager)
- OpenAI API key

## Step 1: Install Poetry

### macOS/Linux
```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

### Windows (PowerShell)
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

Verify installation:
```bash
poetry --version
```

## Step 2: Clone Repository and Setup Environment

```bash
cd Bhanusatish8
cp .env.example .env
```

Edit `.env` with your OpenAI API key:
```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

## Step 3: Start PostgreSQL with pgVector

### Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

Wait for the database to be ready (check health):
```bash
docker-compose ps
```

Verify database is running:
```bash
docker exec agentic_ai_postgres psql -U postgres -d agentic_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Manual PostgreSQL Setup (Alternative)

If not using Docker, install PostgreSQL 14+ and pgVector:

1. Install PostgreSQL: https://www.postgresql.org/download/
2. Install pgVector extension: https://github.com/pgvector/pgvector#installation
3. Create database:
   ```bash
   createdb agentic_ai_db
   psql agentic_ai_db -c "CREATE EXTENSION vector;"
   ```

## Step 4: Create Virtual Environment with Poetry

```bash
# Create virtual environment and install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

This will install:
- FastAPI & Uvicorn
- SQLAlchemy & psycopg2 (PostgreSQL driver)
- pgVector Python bindings
- OpenAI SDK
- All other dependencies specified in `pyproject.toml`

## Step 5: Initialize Database

The database will be initialized automatically when the application starts, but you can run it manually:

```bash
python -c "from database import init_db; init_db()"
```

You should see: `✓ Database initialized successfully`

## Step 6: Run the Application

```bash
python app_updated.py
```

Or with Uvicorn directly:
```bash
poetry run uvicorn app_updated:app --host 0.0.0.0 --port 17000 --reload
```

Expected output:
```
✓ Database initialized
✓ Application started successfully
INFO:     Application startup complete [uvicorn]
INFO:     Uvicorn running on http://0.0.0.0:17000
```

## Step 7: Test the Application

### Health Check
```bash
curl http://localhost:17000/health
```

Expected response:
```json
{
  "status": "ok",
  "pgvector_enabled": true,
  "embedding_model": "text-embedding-3-small",
  "chat_model": "gpt-3.5-turbo"
}
```

### Ingest a Document
```bash
curl -X POST http://localhost:17000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "urn": "doc-001",
    "text": "Machine learning is a subset of artificial intelligence that focuses on data-driven algorithms.",
    "metadata": {"source": "education", "topic": "ML"}
  }'
```

Response:
```json
{
  "status": "ingested",
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "urn": "doc-001"
}
```

### Query Documents
```bash
curl -X POST http://localhost:17000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 3
  }'
```

Response:
```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "retrieved": [
    {
      "doc_id": "550e8400-e29b-41d4-a716-446655440000",
      "urn": "doc-001",
      "score": 0.8234
    }
  ]
}
```

### Get Statistics
```bash
curl http://localhost:17000/stats
```

Response:
```json
{
  "total_documents": 1,
  "total_queries": 1,
  "total_agent_logs": 0,
  "embedding_model": "text-embedding-3-small",
  "embedding_dimension": 1536
}
```

### List Documents
```bash
curl http://localhost:17000/documents?limit=10
```

## Development Workflow

### Format Code
```bash
poetry run black .
poetry run isort .
```

### Run Linting
```bash
poetry run flake8 .
poetry run mypy .
```

### Run Tests
```bash
poetry run pytest
```

## Troubleshooting

### Issue: `OPENAI_API_KEY not set`
**Solution:** Make sure you've created a `.env` file with your actual OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your API key
```

### Issue: `Cannot connect to database`
**Solution:** Verify PostgreSQL is running:
```bash
# If using Docker Compose:
docker-compose ps

# If running locally:
psql -U postgres -d agentic_ai_db -c "SELECT 1;"
```

### Issue: `pgvector extension not found`
**Solution:** Create the extension:
```bash
docker exec agentic_ai_postgres psql -U postgres -d agentic_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Issue: Python version mismatch
**Solution:** Poetry uses the Python version specified in `pyproject.toml`. Install Python 3.10+:
```bash
python --version  # Should be 3.10 or higher
poetry install --no-cache
```

### Issue: Dependencies not installing
**Solution:** Update Poetry and clear cache:
```bash
poetry self update
poetry cache clear . --all
poetry install
```

## Project Structure

```
.
├── app_updated.py          # Main FastAPI application
├── database.py             # SQLAlchemy models and DB config
├── vector_store.py         # pgVector store implementation
├── pyproject.toml          # Poetry dependency configuration
├── docker-compose.yml      # PostgreSQL + pgVector setup
├── init.sql                # Database initialization script
├── .env.example            # Environment variables template
├── requirements.txt        # Backup pip requirements (legacy)
└── SETUP_GUIDE.md         # This file
```

## Performance Tips

1. **Embedding Caching:** Cache frequently used embeddings to reduce API calls
2. **Database Indexes:** pgVector indexes are created automatically for fast vector searches
3. **Batch Operations:** Use batch ingestion for multiple documents
4. **Connection Pooling:** SQLAlchemy manages connection pooling automatically

## Security Considerations

- Never commit `.env` file (it's in `.gitignore`)
- Use strong database passwords in production
- Implement authentication for API endpoints
- Monitor OpenAI API usage and costs

## Next Steps

1. Customize the embedding model if needed (try `text-embedding-3-large`)
2. Add authentication to API endpoints
3. Implement document deletion and updates
4. Add batch ingestion endpoint
5. Create admin dashboard for document management
6. Set up monitoring and logging

## Support

For issues with:
- **Poetry:** https://python-poetry.org/docs/
- **FastAPI:** https://fastapi.tiangolo.com/
- **pgVector:** https://github.com/pgvector/pgvector
- **SQLAlchemy:** https://docs.sqlalchemy.org/

## License

MIT License - See LICENSE file for details
