from app.services.preprocessing import preprocess_dataset
from app.utils.chunker import chunk_job_descriptions
from app.services.embeddings import embed_chunks
from app.utils.bm25 import create_bm25_retriever
from app.utils.vector_store import get_qdrant_client, populate_vectordb

raw_data_path = Path(settings.raw_data_dir) / settings.file_name
processed_data_path = Path(settings.processed_data_dir) / settings.processed_file_name
chunked_data_path = Path(settings.chunked_data_dir) / settings.chunked_file_name
embeddings_data_path = Path(settings.embeddings_data_dir) / settings.embeddings_file_name

keyword_retriever_path = Path(settings.keyword_retriever_dir) / settings.keyword_retriever_file

# 1. Preprocess the dataset and store it as json
if not processed_data_path.exists():
    print('\nProcessing the data...')
    preprocess_dataset(raw_data_path = raw_data_path, processed_data_path=processed_data_path)

# 2. Chunk the data 
if not chunked_data_path.exists():
    print('\nCreating chunks...')
    chunk_job_descriptions(processed_data_path=processed_data_path, chunk_path=chunked_data_path)

# 3. Generate embeddings
if not embeddings_data_path.exists():
    print('Genrating embeddings')
    embed_chunks(chunk_path=chunked_data_path, embed_path=embeddings_data_path)

# 4. Populate keyword retriever
if not keyword_retriever_path.exists():
    print('Initailizing keyword retriever')
    create_bm25_retriever(chunks_path=chunked_data_path, storage_path=keyword_retriever_path)

# 5. Populate vector database
client = get_qdrant_client()
if not client.collection_exists(collection_name=settings.vector_db_collection_name):
    print('\nPopulating vector database...')
    populate_vectordb(client, embeddings_path=embeddings_data_path, chunks_path=chunked_data_path)