"""
FastAPI application for RAG Document Intelligence.
Endpoints: /ingest, /query, /evaluate, /health
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker, ChunkConfig
from src.ingestion.embedder import DocumentEmbedder
from src.vectorstore.pgvector_store import PGVectorStore
from src.agents.orchestrator import RAGOrchestrator
from src.evaluation.harness import EvaluationHarness


# ---------------------------------------------------------------------------
# Globals (initialized at startup)
# ---------------------------------------------------------------------------

vectorstore: Optional[PGVectorStore] = None
orchestrator: Optional[RAGOrchestrator] = None
evaluator: Optional[EvaluationHarness] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global vectorstore, orchestrator, evaluator
    vectorstore = PGVectorStore(connection_string=os.environ["PGVECTOR_URL"])
    orchestrator = RAGOrchestrator(vectorstore=vectorstore)
    evaluator = EvaluationHarness()
    yield
    # Cleanup on shutdown
    vectorstore.close()


app = FastAPI(
    title="RAG Document Intelligence API",
    description="Production RAG pipeline with multi-agent orchestration",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question")
    stream: bool = Field(False, description="Stream response token-by-token")


class QueryResponse(BaseModel):
    answer: str
    latency_ms: float
    sources: List[dict] = []


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    document_count: int


class EvalRequest(BaseModel):
    questions: List[str]
    ground_truths: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """Ingest PDF/DOCX/TXT files into the vector store."""
    loader = DocumentLoader()
    chunker = DocumentChunker(ChunkConfig(strategy="semantic"))
    embedder = DocumentEmbedder()

    all_docs = []
    for file in files:
        content = await file.read()
        docs = loader.load_bytes(content, filename=file.filename)
        all_docs.extend(docs)

    chunks = chunker.chunk(all_docs)
    embeddings = embedder.embed(chunks)
    vectorstore.add_documents(chunks, embeddings)

    return IngestResponse(
        status="success",
        chunks_indexed=len(chunks),
        document_count=len(all_docs),
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG pipeline."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    start = time.perf_counter()
    answer = orchestrator.query(request.question)
    latency_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(answer=answer, latency_ms=round(latency_ms, 2))


@app.post("/evaluate")
async def evaluate_pipeline(request: EvalRequest):
    """Run RAGAS evaluation on the pipeline."""
    results = evaluator.run(
        questions=request.questions,
        orchestrator=orchestrator,
        ground_truths=request.ground_truths,
    )
    return results
