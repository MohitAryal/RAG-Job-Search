from fastapi import FastAPI
from models import QueryRequest, QueryResponse, SearchResult, JobLocation
from datetime import datetime
from app.inference_pipeline import run_pipeline

app = FastAPI(
    title="Job Search RAG API",
    description="Hybrid search + reranking + LLM-enriched results for job search",
)


@app.post("/api/query", response_model=QueryResponse)
async def query_jobs(request: QueryRequest):
    
    results = run_pipeline(**request.dict())

    response = QueryResponse(
        answer=f"Found {len(results)} jobs relevant to your query: '{request.query}'.",
        results=results,
        timestamp=datetime.now(),
    )

    return response