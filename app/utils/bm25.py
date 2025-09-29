import pickle
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import re
import pandas as pd
from app.config import settings
from app.config import logger

chunked_data_path = Path(settings.chunked_data_dir) / settings.chunked_file_name
keyword_retriever_path = Path(settings.keyword_retriever_dir) / settings.keyword_retriever_file

def preprocess_text_for_bm25(text: str) -> List[str]:
    """
    Preprocess text for BM25 by tokenizing and cleaning.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split by whitespace and filter empty strings
    tokens = [token for token in text.split() if token and len(token) > 1]
    
    return tokens


def create_bm25_retriever() -> BM25Okapi:
    """
    Create BM25 retriever from chunks and save to persistent storage.
    
    Args:
        chunks: List of chunk contents
        storage_path: Path to save the BM25 model
    """
    # Get chunks from path
    df = pd.read_json(chunks_path)
    chunks = df['content'].to_list()

    # Extract and preprocess text content from chunks
    corpus = []
    
    for chunk in chunks:
        # Preprocess text for BM25
        tokens = preprocess_text_for_bm25(chunk)
        corpus.append(tokens)
    
    # Create BM25 retriever
    bm25_retriever = BM25Okapi(
        corpus, 
        k1=settings.bm25_k1,
        b=settings.bm25_b  
    )
    
    # Save to persistent storage
    with open(storage_path, 'wb') as f:
        pickle.dump(bm25_retriever, f)
    
    logger.info(f"BM25 retriever saved to {storage_path}")
    return