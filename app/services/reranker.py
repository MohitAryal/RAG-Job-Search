import json
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import CrossEncoder
from app.config import settings
from pathlib import Path

file_path = Path(settings.processed_data_dir) / settings.processed_file_name


def load_jobs():
    with open(file_path, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    return processed_data


def extract_jobs_by_ids(job_ids: List[str], data_dict) -> List[Dict[str, Any]]:
    """Extract specific jobs from processed data using job IDs."""
    
    job_ids_set = set(job_ids)
    extracted_jobs = [job for job in data_dict if job['ID'] in job_ids_set]
    
    return extracted_jobs


def prepare_job_text(job: Dict[str, Any]) -> str:
    """Prepare job text for cross-encoding by combining relevant fields."""
    text_parts = []
    
    # Job title and category
    if job.get('Job Title'):
        text_parts.append(f"Title: {job['Job Title']}")
    
    if job.get('Job Category'):
        text_parts.append(f"Category: {job['Job Category']}")
    
    # Company and location
    if job.get('Company Name'):
        text_parts.append(f"Company: {job['Company Name']}")
    
    if job.get('Job Location'):
        location_info = job['Job Location']
        location_parts = []
        if location_info.get('cities'):
            location_parts.extend(location_info['cities'])
        if location_info.get('states'):
            location_parts.extend(location_info['states'])
        if location_info.get('countries'):
            location_parts.extend(location_info['countries'])
        if location_parts:
            text_parts.append(f"Location: {', '.join(location_parts)}")
        if location_info.get('is_remote'):
            text_parts.append("Remote: Yes")
    
    # Job level and description
    if job.get('Job Level'):
        text_parts.append(f"Level: {job['Job Level']}")
    
    if job.get('Job Description'):
        # Clean and truncate description
        description = job['Job Description'].replace('\n', ' ').replace('**', '').strip()
        text_parts.append(f"Description: {description}")
    
    # Tags
    if job.get('Tags') and job['Tags'] != ['nan']:
        tags = [tag for tag in job['Tags'] if tag != 'nan']
        if tags:
            text_parts.append(f"Tags: {', '.join(tags)}")
    
    return " | ".join(text_parts)

cross_encoder = CrossEncoder(settings.reranker_model)

def rerank_jobs(job_ids: List[str], query: str, top_k:int) -> List[Dict[str, Any]]:
    """Rerank jobs using cross-encoder based on query relevance."""  
    # Get all the jobs
    data = load_jobs()

    # Get the jobs
    jobs = extract_jobs_by_ids(job_ids, data)

    # Prepare job texts
    job_texts = [prepare_job_text(job) for job in jobs]
    
    # Create query-document pairs for cross-encoder
    pairs = [(query, job_text) for job_text in job_texts]
    
    # Get relevance scores from cross-encoder
    scores = cross_encoder.predict(pairs)
    
    # Combine jobs with scores and sort by relevance
    job_score_pairs = list(zip(jobs, scores))
    job_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k results
    return [job for job, score in job_score_pairs[:top_k]]