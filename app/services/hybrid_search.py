from app.services.keyword_retriever import search_bm25
from app.services.vector_retriever import qdrant_semantic_search
from app.config import settings
from typing import List


def perform_hybrid_search(query: str) -> List[str]:
    '''
    Performs keyword + semantic search and returns top n document ids.
    
    Args:
    'query': The query string.
    'filter': Any filters to be applied to the output.

    Result:
    List of document ids
    '''
    
    # Get the result from keyword retriever
    bm25_result = search_bm25(query)

    # Get the result from vector retriever
    qdrant_result = qdrant_semantic_search(query)

    # Get job ids appearing in both result
    all_jobs = set(bm25_result.keys()) & set(qdrant_result.keys())

    # Compute RRF scores
    fused_scores = {}
    for job in all_jobs:
        rrf_score = (1 / (settings.k + bm25_result[job])) + (1 / (settings.k + qdrant_result[job]))
        fused_scores[job] = rrf_score

    # Sort by fused score descending
    ranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:settings.default_top_k]
    job_ids = [job for job, _ in ranked_results]

    return job_ids