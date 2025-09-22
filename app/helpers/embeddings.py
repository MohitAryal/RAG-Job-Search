from sentence_transformers import SentenceTransformer
from app.config import settings

embedder = SentenceTransformer(settings.embedding_model)

def embed_chunks(chunks: int) :
    embedded = embedder.encode(chunks)
    return embedded