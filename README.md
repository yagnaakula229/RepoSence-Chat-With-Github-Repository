# RepoSense AI – Chat with Any GitHub Repository

A full-stack project for ingesting GitHub repositories and querying them through a Retrieval-Augmented Generation (RAG) pipeline.

## Backend

The backend is built with FastAPI and exposes two endpoints:

- `POST /api/ingest-repo` — ingest a GitHub repository and index supported files
- `POST /api/query` — query the indexed repository and return an answer with source references

### Run the backend locally

1. Navigate to the backend folder:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file from `.env.example` and add your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Environment variables

- `OPENAI_API_KEY` — optional; if set, the app uses OpenAI for embeddings and LLM generation.
- `HUGGINGFACEHUB_API_TOKEN` — optional; used for Hugging Face model inference when no OpenAI key is configured.
- `EMBEDDING_MODEL` — optional, default `text-embedding-3-small`
- `LLM_MODEL` — optional, default `gpt-3.5-turbo` for OpenAI or `google/flan-t5-small` / `distilgpt2` for Hugging Face.
- `GITHUB_TOKEN` — optional, improves GitHub API rate limits

## Project structure

- `backend/main.py` — FastAPI application entrypoint
- `backend/routes/` — API routes for ingestion and queries
- `backend/services/` — GitHub loader, embeddings, and RAG pipeline logic
- `backend/utils/` — shared utility helpers
- `frontend/` — React frontend application

## Frontend

The frontend is a Vite-powered React app that connects to backend API endpoints via `/api`.

### Run the frontend locally

1. Navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Open the local URL shown by Vite (usually `http://localhost:5173`).

The frontend proxy sends `/api` requests to `http://localhost:8000`, so start the backend server first.
