from qdrant_client import QdrantClient
from app.config import settings

qdrant_client = QdrantClient(
    url=settings.vector_db_url,
    api_key=settings.vector_db_api_key,
)

print(qdrant_client.get_collections())