from fastapi import FastAPI
from app.models import QueryRequest
from app.inference_pipeline import run_pipeline
from app.config import logger

router = FastAPI(
    title="Job Search RAG API",
    description="Hybrid search + reranking + LLM-enriched results for job search",
)


@router.post("/api/query", response_model=str)
def query_jobs(request: QueryRequest):
    
    logger.info('Sending request to pipeline')
    results = run_pipeline(request.query)

    logger.info('Results found Returning them')
    return results