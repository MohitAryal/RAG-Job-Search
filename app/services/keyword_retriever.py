import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from app.utils.bm25 import preprocess_text_for_bm25
import pandas as pd
from app.config import settings
from pathlib import Path

storage_path = Path(settings.keyword_retriever_dir) / settings.keyword_retriever_file
chunk_path = Path(settings.chunked_data_dir) / settings.chunked_file_name


def search_bm25(query: str, top_k: int = 100) -> List[Dict[str, Any]]:
    """
    Search using BM25 retriever.
    
    Args:
        query: Search query string
        top_k: Number of top results to return
        
    Returns:
        List of retrieved job ids with their ranks
    """

    # Load the model from the pickle file
    with open(storage_path, 'rb') as f:
        bm25_retriever = pickle.load(f)

    # Preprocess query
    query_tokens = preprocess_text_for_bm25(query)
    
    # Get BM25 scores for all documents
    scores = bm25_retriever.get_scores(query_tokens)
    
    # Get top-k results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = get_doc_ids(ids=top_indices, scores=scores)
    
    return results


def get_doc_ids(ids: List[int], scores: List[float]) -> Dict:
    """
    Extracts the most relevant document score per unique job ID from a JSON chunk file.

    Args:
    ids : List[int]
        List of row indices pointing to relevant chunks in the JSON file.
    
    scores : List[float]
        Relevance scores corresponding to each index in `ids`.

    Returns:
    Dict[str, float]
        A dictionary where keys are unique job_id and values are the rank of each job based on average score attained for the job.
    """

    # Read the chunks from json file
    chunks = pd.read_json(chunk_path)

    # Get only the relevant chunks based on ids
    relevant_chunks = chunks.iloc[ids]
    jobs = relevant_chunks.copy()

    # Get job ids for each relevant chunk
    jobs['job_id'] = jobs['metadata'].apply(lambda x: x.get('job_id'))

    # Get score corresponding to each job id
    jobs['scores'] = [scores[i] for i in ids]

    # Store unique job ids with their maximum score and sort by scores
    job_scores = jobs.groupby('job_id')['scores'].mean().sort_values(ascending=False)

    # Get a dictionary consisting jobs ids with their rank
    job_rank = {job_id: rank + 1 for rank, job_id in enumerate(job_scores.index)}

    return job_rank