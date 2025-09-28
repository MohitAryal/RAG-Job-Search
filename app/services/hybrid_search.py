from app.services.keyword_retriever import search_bm25
from app.services.vector_retriever import qdrant_semantic_search
from app.config import settings
from typing import List
import pandas as pd


def perform_hybrid_search(query: str) -> List[str]:
    '''
    Performs keyword + semantic search and returns top n document ids.
    
    Args:
    'query': The query string.
    'filter': Any filters to be applied to the output.

    Result:
    List of document ids
    Dictionary of contents corresponding to each job id
    '''
    
    # Get the result from keyword retriever
    bm25_result = search_bm25(query)

    # Get the result from vector retriever
    qdrant_result = qdrant_semantic_search(query)

    joined = pd.merge(bm25_result, qdrant_result, on='job_id', suffixes=('_bm', '_qd'))

    joined['rrf'] = (1 / (settings.k + joined['rank_bm'])) + (1 / (settings.k + joined['rank_qd']))

    joined.sort_values(by='rrf', ascending=False, inplace=True)

    top_k = joined.iloc[:15].copy()

    top_k['combined_contents'] = top_k.apply(
    lambda row: list(set(row['content_bm'] + row['content_qd'])),
    axis=1
)
    
    job_ids = top_k['job_id'].to_list()
    contents = dict(zip(top_k['job_id'], top_k['combined_contents']))
    return job_ids, contents