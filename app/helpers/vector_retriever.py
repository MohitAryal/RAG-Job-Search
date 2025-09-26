from app.utils.vector_store import get_qdrant_client
from app.utils.embedding_function import embed_function
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from typing import List, Dict, Any, Optional
from app.config import settings
from collections import defaultdict


def qdrant_semantic_search(query: 'str', filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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

    # Build filter conditions if provided
    filter_conditions = None
    if filters:
        filter_conditions = _build_filter_conditions(filters)
    
    # Perform the search
    search_results = client.query_points(
        collection_name=settings.vector_db_collection_name,
        query=query_embedding,
        limit=100,
        query_filter=filter_conditions,
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


def _build_filter_conditions(filters: Dict[str, Any]) -> Filter:
    """
    Build Qdrant filter conditions from filter dictionary
    
    Args:
        filters: Dictionary of filter conditions
        
    Returns:
        Qdrant Filter object
    """
    conditions = []
    
    for field, value in filters.items():
        if isinstance(value, dict):
            # Handle range queries (gte, lte, gt, lt)
            if any(op in value for op in ['gte', 'lte', 'gt', 'lt']):
                range_condition = Range()
                if 'gte' in value:
                    range_condition.gte = value['gte']
                if 'lte' in value:
                    range_condition.lte = value['lte']
                if 'gt' in value:
                    range_condition.gt = value['gt']
                if 'lt' in value:
                    range_condition.lt = value['lt']
                
                conditions.append(
                    FieldCondition(key=field, range=range_condition)
                )
            
            # Handle match any (OR conditions)
            elif 'any' in value:
                conditions.append(
                    FieldCondition(
                        key=field, 
                        match=MatchValue(any=value['any'])
                    )
                )
            
            # Handle exact match within dict
            elif 'value' in value:
                conditions.append(
                    FieldCondition(
                        key=field, 
                        match=MatchValue(value=value['value'])
                    )
                )
        
        elif isinstance(value, list):
            # Handle list of values (match any)
            conditions.append(
                FieldCondition(
                    key=field, 
                    match=MatchValue(any=value)
                )
            )
        
        else:
            # Handle single value exact match
            conditions.append(
                FieldCondition(
                    key=field, 
                    match=MatchValue(value=value)
                )
            )
    
    return Filter(must=conditions) if conditions else None