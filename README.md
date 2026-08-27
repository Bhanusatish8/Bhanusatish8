# Agentic AI with pgVector 🤖

An advanced **Retrieval-Augmented Generation (RAG)** system built with FastAPI, OpenAI embeddings, and PostgreSQL + pgVector for intelligent document retrieval and question answering.

## 🎯 Features

- ✅ **Semantic Search** - Find documents using AI embeddings
- ✅ **LLM Integration** - OpenAI GPT-3.5 Turbo for intelligent responses
- ✅ **Vector Database** - PostgreSQL with pgVector for efficient similarity search
- ✅ **Production-Ready** - Async FastAPI with full error handling
- ✅ **Dependency Management** - Poetry for reproducible builds
- ✅ **Docker Support** - One-command database setup
- ✅ **Comprehensive Logging** - Track queries and agent requests

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- OpenAI API key

### 1. Clone Repository
```bash
git clone https://github.com/Bhanusatish8/Bhanusatish8.git
cd Bhanusatish8
```

### 2. Setup Environment
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 3. Start PostgreSQL with pgVector
```bash
docker-compose up -d
```

### 4. Install Dependencies with Poetry
```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate virtual environment
poetry shell
```

### 5. Run Application
```bash
python app_updated.py
```

Application will start at `http://0.0.0.0:17000`

## 📚 API Endpoints

### Health Check
```bash
curl http://localhost:17000/health
```

### Ingest Document
```bash
curl -X POST http://localhost:17000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "urn": "doc-001",
    "text": "Machine learning is a subset of AI",
    "metadata": {"source": "wiki"}
  }'
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

### Get Statistics
```bash
curl http://localhost:17000/stats
```

### List Documents
```bash
curl http://localhost:17000/documents?limit=10
```

## 📁 Project Structure

```
.
├── app_updated.py          # Main FastAPI application
├── database.py             # SQLAlchemy ORM models
├── vector_store.py         # pgVector store implementation
├── pyproject.toml          # Poetry dependency configuration
├── docker-compose.yml      # PostgreSQL + pgVector setup
├── init.sql                # Database initialization schema
├── requirements.txt        # Pip requirements (backup)
├── .env.example            # Environment configuration template
├── SETUP_GUIDE.md          # Detailed setup documentation
└── README.md              # This file
```

## 🔧 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | FastAPI | Async REST API |
| **Web Server** | Uvicorn | ASGI application server |
| **Database** | PostgreSQL | Data persistence |
| **Vector Store** | pgVector | Semantic search |
| **Embeddings** | OpenAI | Text embeddings (1536-dim) |
| **LLM** | GPT-3.5 Turbo | Question answering |
| **ORM** | SQLAlchemy | Database abstraction |
| **Dependency Management** | Poetry | Reproducible builds |

## 🗄️ Database Schema

### Documents Table
```sql
- id: UUID (primary key)
- urn: String (document identifier)
- text: Text (document content)
- embedding: Vector(1536) (OpenAI embeddings)
- metadata: JSONB (custom metadata)
- created_at: Timestamp
- updated_at: Timestamp
```

### Query History Table
```sql
- id: UUID (primary key)
- query: Text (user query)
- embedding: Vector(1536) (query embedding)
- answer: Text (LLM response)
- retrieved_docs: JSONB (retrieved documents)
- created_at: Timestamp
```

### Agent Logs Table
```sql
- id: UUID (primary key)
- request_data: JSONB (agent request)
- response_data: JSONB (agent response)
- status: String (success/failure)
- error_message: Text (error details)
- created_at: Timestamp
```

## 🔄 How It Works

1. **Document Ingestion**
   - User submits document with URN and text
   - System generates OpenAI embedding (1536 dimensions)
   - Document and embedding stored in PostgreSQL

2. **Query Processing**
   - User submits query
   - Query is embedded using OpenAI
   - pgVector performs cosine similarity search
   - Top-k similar documents retrieved

3. **Response Generation**
   - Retrieved documents used as context
   - Context + query sent to GPT-3.5 Turbo
   - LLM generates contextual response
   - Query and response logged for audit trail

## 📊 Performance Characteristics

- **Embedding Generation**: ~100-500ms per document (OpenAI API)
- **Vector Search**: <10ms for similarity search (pgVector index)
- **Response Generation**: 1-3 seconds (OpenAI API)
- **Storage**: ~1KB per document + embedding vectors

## 🔒 Security Considerations

- Never commit `.env` file (add to `.gitignore`)
- Use strong passwords for production databases
- Implement API authentication for production
- Monitor OpenAI API costs and usage
- Use connection pooling for database
- Set rate limits on API endpoints

## 🐛 Troubleshooting

### Database Connection Error
```bash
# Verify PostgreSQL is running
docker-compose ps

# Check database exists
docker exec agentic_ai_postgres psql -U postgres -l
```

### pgVector Extension Not Found
```bash
# Install extension
docker exec agentic_ai_postgres psql -U postgres -d agentic_ai_db \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Poetry Dependency Conflict
```bash
# Clear cache and reinstall
poetry cache clear . --all
poetry install --no-cache
```

### OpenAI API Error
- Verify API key is set in `.env`
- Check API key has sufficient credits
- Review rate limits and quotas

## 📖 Detailed Documentation

See [SETUP_GUIDE.md](./SETUP_GUIDE.md) for comprehensive setup instructions and examples.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📜 License

MIT License - See LICENSE file for details

## 📞 Support

For issues with:
- **FastAPI**: https://fastapi.tiangolo.com/
- **Poetry**: https://python-poetry.org/docs/
- **pgVector**: https://github.com/pgvector/pgvector
- **OpenAI**: https://platform.openai.com/docs

---

**Built with ❤️ by Bhanusatish8**
