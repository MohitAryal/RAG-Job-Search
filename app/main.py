from fastapi import FastAPI
from models import QueryRequest, QueryResponse
from app.inference_pipeline import run_pipeline

app = FastAPI(
    title="Job Search RAG API",
    description="Hybrid search + reranking + LLM-enriched results for job search",
)


@app.post("/api/query", response_model=QueryResponse)
def query_jobs(request: QueryRequest):
    
    results = run_pipeline(request)
    response = QueryResponse(results=results)

    return response