from app.services.hybrid_search import perform_hybrid_search
from app.services.reranker import rerank_jobs
from app.services.LLM_integration import llm_result


def run_pipeline(query: str, top_k:int):
    # 1. Search for relevant documents
    print('retrieving')
    search_result = perform_hybrid_search(query=query)

    # 2. Rerank the search results and retrieve top k results
    print('reranking')
    reranked_result = rerank_jobs(job_ids=search_result, query=query, top_k=top_k)

    # 3. Call LLM to enrich the results
    print('llm')
    llm_output = llm_result(reranked_result, query)

    return llm_output

run_pipeline(query='ML engineer with rag langchain knowledge and 4 years of experience.', top_k=5)