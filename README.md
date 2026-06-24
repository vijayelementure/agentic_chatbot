# Agentic RAG — Google Drive + Website Knowledge Bot

An agentic Retrieval-Augmented Generation service that:

- Reads documents from a **Google Drive** folder (Docs, Sheets, Slides, PDFs, DOCX, TXT)
- Crawls and indexes a target **website** (default: `https://capitalnxt.co.in/`)
- Stores embeddings in a local **ChromaDB** vector store using **Gemini** embeddings
- Runs an **agentic loop** (Gemini function-calling): the model decides when and
  what to search for, can issue multiple searches, and grounds answers in
  retrieved content with citations
- Ships as a proper Python package with a **FastAPI** service, CLI, tests,
  Docker image, and CI pipeline

## Architecture

```
                ┌──────────────┐      ┌──────────────────┐
                │ Google Drive │      │  Target Website   │
                └──────┬───────┘      └────────┬─────────┘
                       │ DriveReader            │ WebsiteCrawler
                       └──────────┬─────────────┘
                                  ▼
                          chunk_text()
                                  ▼
                    GeminiEmbedder → VectorStore (ChromaDB)
                                  ▲
                                  │ search_knowledge_base tool
                       ┌──────────┴──────────┐
                       │   AgenticRAG agent   │  ← Gemini function calling loop
                       └──────────┬──────────┘
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
              CLI (cli.py)   FastAPI (api/)   Python import
```

## Project layout

```
agentic_rag/
├── agentic_rag/                 # the installable package
│   ├── __init__.py
│   ├── settings.py               # typed, validated config (pydantic-settings)
│   ├── logging_config.py         # centralized logging setup
│   ├── exceptions.py             # app-specific exception hierarchy
│   ├── chunking.py                # text chunking
│   ├── drive_reader.py           # Google Drive ingestion (retried, typed)
│   ├── web_crawler.py            # website crawler (retried, typed)
│   ├── vector_store.py           # ChromaDB + Gemini embeddings wrapper
│   ├── ingest.py                  # ingestion pipeline (importable + CLI)
│   ├── agent.py                   # agentic loop (Gemini function calling)
│   ├── cli.py                     # interactive CLI chat
│   └── api/
│       ├── main.py                # FastAPI app factory + lifespan
│       ├── dependencies.py        # DI: agent singleton, optional API-key auth
│       ├── schemas.py             # request/response Pydantic models
│       ├── routes_health.py       # GET /health
│       ├── routes_ask.py          # POST /ask
│       └── routes_ingest.py       # POST /ingest, GET /ingest/status
├── tests/                          # pytest suite (unit + API tests, all mocked)
├── .github/workflows/ci.yml        # lint + test + docker build on push/PR
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt / requirements-dev.txt
├── .env.example
└── .gitignore
```

## 1. Setup

```bash
cd agentic_rag
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
# or, for development (tests/lint included):
pip install -r requirements-dev.txt
```

## 2. Configure

```bash
cp .env.example .env
```

Edit `.env` — key variables:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Get one free at https://aistudio.google.com/apikey |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | Path to a service-account JSON key (see below) |
| `GOOGLE_DRIVE_FOLDER_ID` | The folder ID to ingest (from the Drive folder URL) |
| `WEBSITE_URL` | Defaults to `https://capitalnxt.co.in/` |
| `MAX_CRAWL_PAGES` | Crawl limit (default 50) |
| `API_KEY` | Optional — if set, `/ask` and `/ingest` require `X-API-Key` header |

All settings are validated at startup via `pydantic-settings` — missing/invalid
config fails fast with a clear error instead of a confusing runtime crash.

### Getting a Google Drive service account (read-only)

1. Go to https://console.cloud.google.com/ → create/select a project.
2. Enable the **Google Drive API** (APIs & Services → Library).
3. Create a **Service Account** (IAM & Admin → Service Accounts), then a JSON key
   for it — download it and save as `service_account.json` in this folder
   (or point `GOOGLE_SERVICE_ACCOUNT_FILE` at it).
4. Open the target folder in Google Drive, click **Share**, and share it with
   the service account's email address (`xxx@xxx.iam.gserviceaccount.com`) —
   Viewer access is enough.
5. Copy the folder ID from the folder's URL into `GOOGLE_DRIVE_FOLDER_ID`.

## 3. Ingest content

Pulls Drive docs + crawls the website, chunks, embeds (Gemini), and stores
everything in `./data/chroma_db`. Safe to re-run any time (upserts):

```bash
make ingest
# or: python -m agentic_rag.ingest
```

## 4. Use the bot

**CLI:**
```bash
make chat
# or: python -m agentic_rag.cli
```

**FastAPI service** (interactive docs at `/docs`):
```bash
make serve
# or: uvicorn agentic_rag.api.main:app --host 0.0.0.0 --port 8000 --reload
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness/readiness + chunk count + version |
| `POST` | `/ask` | `{"question": "...", "max_tool_calls": 5}` → `{"question", "answer"}` |
| `POST` | `/ingest` | Triggers a fresh Drive + website ingestion in the background |
| `GET` | `/ingest/status` | Status of the last ingestion run |

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What services does CapitalNxt offer?"}'
```

If `API_KEY` is set in `.env`, add `-H "X-API-Key: <your-key>"` to protected calls.

**Docker:**
```bash
docker compose up --build
```

## How the agent works

1. The user's question goes to Gemini along with a `search_knowledge_base` tool.
2. Gemini decides if/when to call the tool — it can call it multiple times with
   different queries to gather enough context (e.g. one search for Drive docs,
   another for website content), bounded by `MAX_TOOL_CALLS`.
3. Each tool call queries the Chroma vector store (Gemini embeddings, cosine
   similarity) and returns the top matching chunks with their source.
4. Gemini synthesizes a final answer strictly grounded in retrieved chunks,
   citing sources.

## Testing & quality

```bash
make test     # pytest with coverage
make lint     # ruff
make format   # ruff --fix
```

The test suite mocks all external calls (Gemini, Drive API, HTTP requests) so
it runs offline and deterministically. CI (`.github/workflows/ci.yml`) runs
lint + tests on Python 3.10/3.11 and verifies the Docker image builds.

## Production notes

- **Vector store**: local file-based Chroma is fine for a single instance.
  For multi-instance deployments, swap in a hosted vector DB (Chroma server,
  Pinecone, Qdrant) by editing `agentic_rag/vector_store.py` — the rest of the
  codebase only depends on the `VectorStore` interface.
- **Retries**: Drive API calls, embedding calls, and HTTP crawl requests all
  use exponential-backoff retries (`tenacity`) for transient failures.
- **Auth**: set `API_KEY` to require an `X-API-Key` header on `/ask` and
  `/ingest`; put a reverse proxy / API gateway in front for TLS and rate limiting.
- **Crawling**: same-domain BFS only; no `robots.txt` parsing is built in —
  add `urllib.robotparser` if strict compliance is required.
- **Secrets**: `GEMINI_API_KEY` and `service_account.json` are loaded from
  the environment / local file at runtime and are never hardcoded or logged.
- **Observability**: structured logs via `LOG_JSON=true`; wire `/health` into
  your orchestrator's readiness probe.
