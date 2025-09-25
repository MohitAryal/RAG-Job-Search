from sentence_transformers import SentenceTransformer
import torch
from app.config import settings

# Use GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model    
embedder = SentenceTransformer(settings.embedding_model, device=device)

def embed_function(text):
    """
    text: List[str] or str
    returns: np.ndarray with shape (n_texts, 384)
    """
    return embedder.encode(
        text,
        batch_size=settings.embedding_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True, 
    )
