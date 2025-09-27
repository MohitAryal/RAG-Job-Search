from app.config import settings
from pathlib import Path
from app.services.hybrid_search import perform_hybrid_search
from app.services.reranker import rerank_jobs


# 1. Get user's query
query='entry level or internship jobs in singapore'

# 7. Search relevant documents
search_result = perform_hybrid_search(query=query)

# 8. Rerank the search results and retrieve top k results
reranked_result = rerank_jobs(job_ids=search_result, query=query)
print(reranked_result)