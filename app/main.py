from fastapi import FastAPI
from app.models import QueryRequest, QueryResponse, MessageResponse
from app.inference_pipeline import run_pipeline
from app.config import logger
from typing import Union

router = FastAPI(
    title="Job Search RAG API",
    description="Hybrid search + reranking + LLM-enriched results for job search",
)


@router.post("/api/query", response_model=Union[QueryResponse, MessageResponse])
def query_jobs(request: QueryRequest):
    
    logger.info('Sending request to pipeline')
    results = run_pipeline(request.query)

    logger.info('Results found Returning them')
    
    if isinstance(results, str):
        # It's a message like "out of scope" or "no match"
        logger.info('Sending message as response')
        return {"message": MessageResponse(llm_response)}
    
    logger.info('Sending result as response')
    return {'results': QueryResponse(results)}