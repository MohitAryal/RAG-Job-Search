from app.services.hybrid_search import perform_hybrid_search
from app.services.reranker import rerank_jobs
from app.services.LLM_integration import llm_result
from typing import Optional
from app.config import settings


def run_pipeline(query: str, top_k:Optional[int] = settings.reranker_top_k):
    # 1. Search for relevant documents
    search_result = perform_hybrid_search(query=query)

    # 2. Rerank the search results and retrieve top k results
    reranked_result = rerank_jobs(job_ids=search_result, query=query, top_k=top_k)

    # 3. Call LLM to enrich the results
    llm_output = llm_result(reranked_result)

    return llm_output