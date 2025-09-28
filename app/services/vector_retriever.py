from app.utils.vector_store import get_qdrant_client
from app.utils.embedding_function import embed_function
from typing import List, Dict, Any
from app.config import settings
from pathlib import Path
from collections import defaultdict
import pandas as pd


chunk_path = Path(settings.chunked_data_dir) / settings.chunked_file_name

chunks_data = pd.read_json(chunk_path)


def qdrant_semantic_search(query: str) -> pd.DataFrame:
    """
    Perform semantic search in Qdrant collection
    
    Args:
        query: Query string for semantic search
        
    Returns:
        DataFrame with columns: job_id, rank, content
        where content is a list of chunks for each job_id
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

    # Accumulate scores and point IDs per job_id
    job_data = defaultdict(lambda: {'scores': [], 'point_ids': []})

    for point in points:
        job_id = point.payload['job_id']
        score = point.score
        point_id = point.id
        
        job_data[job_id]['scores'].append(score)
        job_data[job_id]['point_ids'].append(point_id)

    # Compute average score per job_id
    job_avg_scores = {}
    for job_id, data in job_data.items():
        job_avg_scores[job_id] = sum(data['scores']) / len(data['scores'])

    # Sort job_ids by average score in descending order
    sorted_jobs = sorted(job_avg_scores.items(), key=lambda item: item[1], reverse=True)

    # Prepare data for DataFrame
    df_data = []
    
    for rank, (job_id, avg_score) in enumerate(sorted_jobs, 1):
        # Get all point IDs for this job
        point_ids = job_data[job_id]['point_ids']
        
        # Retrieve content chunks for all points of this job
        content_chunks = []
        for point_id in point_ids:
            content_chunks.append(str(chunks_data[chunks_data['chunk_id'] == point_id]['content']))
        
        df_data.append({
            'job_id': job_id,
            'rank': rank,
            'content': content_chunks
        })
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    return df