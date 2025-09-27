from app.services.preprocessing import preprocess_dataset
from app.utils.chunker import chunk_job_descriptions
from app.services.embeddings import embed_chunks
from app.utils.bm25 import create_bm25_retriever
from app.utils.vector_store import get_qdrant_client, populate_vectordb

# 1. Preprocess the dataset and store it as json
if not processed_data_path.exists():
    print('\nProcessing the data...')
    preprocess_dataset()

# 2. Chunk the data 
if not chunked_data_path.exists():
    print('\nCreating chunks...')
    chunk_job_descriptions()

# 3. Generate embeddings
if not embeddings_data_path.exists():
    print('Genrating embeddings')
    embed_chunks()

# 4. Populate keyword retriever
if not keyword_retriever_path.exists():
    print('Initailizing keyword retriever')
    create_bm25_retriever()

# 5. Populate vector database
client = get_qdrant_client()
if not client.collection_exists(collection_name=settings.vector_db_collection_name):
    print('\nPopulating vector database...')
    populate_vectordb(client)