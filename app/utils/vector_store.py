from qdrant_client import QdrantClient
from app.config import settings
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

chunks_path = Path(settings.chunked_data_dir) / settings.chunked_file_name
embeddings_path = Path(settings.embeddings_data_dir) / settings.embeddings_file_name


def get_qdrant_client():
    qdrant_client = QdrantClient(
        url=settings.vector_db_url,
        api_key=settings.vector_db_api_key,
        timeout=60.0
    )
    return qdrant_client


def populate_vectordb(
    client,
    batch_size: int = 100
):
    """
    Populates a Qdrant collection with embeddings and metadata in batches.
    """

    embeddings = np.load(embeddings_path).tolist()
    chunks = pd.read_json(chunks_path)
    ids = chunks['chunk_id'].to_list()
    metadatas = chunks['metadata'].to_list()

    # Create collection
    client.recreate_collection(
        collection_name=settings.vector_db_collection_name,
        vectors_config=VectorParams(size=settings.vector_size, distance=Distance.COSINE),
    )

    total = len(embeddings)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_points = [
            PointStruct(
                id=int(ids[i]),
                vector=embeddings[i],
                payload=metadatas[i],
            )
            for i in range(start, end)
        ]
        client.upsert(
            collection_name=settings.vector_db_collection_name,
            points=batch_points,
        )
        print(f"Uploaded batch {start} to {end} ({end - start} vectors)")