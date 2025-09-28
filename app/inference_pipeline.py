from app.services.hybrid_search import perform_hybrid_search
from app.services.reranker import rerank_jobs
from app.services.LLM_integration import llm_result
from app.config import logger


def run_pipeline(query: str):
    # 1. Search for relevant documents
    logger.info('searching')
    ids, contents = perform_hybrid_search(query=query)

    # 2. Rerank the search results and retrieve top k results
    logger.info('reranking')
    reranked_result = rerank_jobs(job_ids=ids, query=query)

    # 3. convert the results from hybrid search and reranking for entry to llm
    keys_to_remove = {"cleaned_title", "Job Description", "Publication Date"}

    for item in reranked_result:
        # Remove unwanted keys
        for key in keys_to_remove:
            item.pop(key)      
        # Assign combined_chunks from hybrid search with size limit
        combined = contents[item["ID"]]
        item["combined_chunks"] = [chunk[:300] for chunk in combined]

    # 3. Call LLM to enrich the results
    logger.info('enriching')
    llm_output = llm_result(reranked_result, query)

    return llm_output
