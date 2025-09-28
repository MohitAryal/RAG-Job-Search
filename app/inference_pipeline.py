from app.services.hybrid_search import perform_hybrid_search
from app.services.reranker import rerank_jobs
from app.services.LLM_integration import llm_result


def run_pipeline(query: str):
    # 1. Search for relevant documents
    search_result = perform_hybrid_search(query=query)

    # 2. Rerank the search results and retrieve top k results
    reranked_result = rerank_jobs(job_ids=search_result, query=query)

    # 3. Call LLM to enrich the results
    llm_output = llm_result(reranked_result, query)

    return llm_output