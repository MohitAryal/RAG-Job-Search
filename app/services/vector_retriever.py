from app.utils.vector_store import get_qdrant_client
from app.utils.embedding_function import embed_function
from typing import List, Dict, Any
from app.config import settings
from collections import defaultdict


def qdrant_semantic_search(query: 'str') -> List[Dict[str, Any]]:
    """
    Perform semantic search in Qdrant collection
    
    Args:
        query_embedding: Query embedding vector
        filters: Metadata filters for search
    
    Returns:
        List of dictionaries with id, score, and rank
    """
    # Get the client
    client = get_qdrant_client()

    # Get embedding for the query
    query_embedding = embed_function(query)
    
    # Perform the search
    search_results = client.query_points(
        collection_name=settings.vector_db_collection_name,
        query=query_embedding,
        limit=100,
    )

    points = search_results.points

    # Accumulate scores per job_id
    job_scores = defaultdict(list)

    for point in points:
        job_id = point.payload['job_id']
        score = point.score
        job_scores[job_id].append(score)

    # Compute average score per job_id
    job_avg_scores = {job_id: sum(scores) / len(scores) for job_id, scores in job_scores.items()}

    # Sort job_ids by average score in descending order
    sorted_jobs = sorted(job_avg_scores.items(), key=lambda item: item[1], reverse=True)

    # Step assign ranks
    ranked_jobs = {job_id: rank + 1 for rank, (job_id, _) in enumerate(sorted_jobs)}
    
    return ranked_jobs