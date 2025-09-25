import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from app.utils.bm25 import preprocess_text_for_bm25
from app.config import settings

top_k = settings.default_top_k


def search_bm25(storage_path, 
                query: str, ) -> List[Dict[str, Any]]:
    """
    Search using BM25 retriever.
    
    Args:
        storage_path: BM25 retriever storage path
        query: Search query string
        top_k: Number of top results to return
        
    Returns:
        List of retrieved chunks with scores
    """

    # Load the model from the pickle file
    with open(storage_path, 'rb') as f:
        bm25_retriever = pickle.load(f)

    print(f"BM25 retriever loaded from {storage_path}")

    # Preprocess query
    query_tokens = preprocess_text_for_bm25(query)
    
    # Get BM25 scores for all documents
    scores = bm25_retriever.get_scores(query_tokens)
    
    # Get top-k results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = []
    for idx in top_indices:
        result = {
            'index': idx,
            'score': float(scores[idx]),
            'rank': len(results) + 1
        }
        results.append(result)
    
    return results