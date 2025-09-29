from app.config import settings
import numpy as np
import pandas as pd
import math
from app.utils.embedding_function import embed_function
from app.config import logger

embedding_dim = settings.embedding_dimension
batch_size = 128
chunk_path = Path(settings.chunked_data_dir) / settings.chunked_file_name
embed_path = Path(settings.embeddings_data_dir) / settings.embeddings_file_name


def embed_chunks() :
    df = pd.read_json(chunk_path)
    chunks = df['content'].to_list()

    total_chunks = len(chunks)
    total_batches = math.ceil(total_chunks / batch_size)
    
    # Create memmap file
    embeddings_memmap = np.lib.format.open_memmap(
        embed_path, mode='w+', dtype=np.float16, shape=(total_chunks, embedding_dim)
    )

    # Loop to write batches of embedded chunks in the memmap
    for i in range(0, total_chunks, batch_size):
        batch_texts = chunks[i:i+batch_size]
        batch_embeddings = embed_function(batch_texts)
        embeddings_memmap[i:i+len(batch_texts)] = batch_embeddings
    
    logger.info('\nEmbeddings complete')
    
