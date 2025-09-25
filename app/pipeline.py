from app.config import settings
from pathlib import Path
from app.helpers.preprocessing import preprocess_dataset
from app.utils.chunker import chunk_job_descriptions

raw_data_path = Path(settings.raw_data_dir) / settings.file_name
processed_data_path = Path(settings.processed_data_dir) / settings.processed_file_name
chunked_data_path = Path(settings.chunked_data_dir) / settings.chunked_file_name
embeddings_file_path = Path(settings.embeddings_data_dir) / settings.embeddings_file_name

# 1. Preprocess the dataset and store it as json
if not processed_data_path.exists():
    preprocess_dataset(raw_data_path = raw_data_path, processed_data_path=processed_data_path)

# 2. Chunk the data 
if not chunked_data_path.exists():
    print(f'\nCreating chunks...')
    chunk_job_descriptions(processed_data_path=processed_data_path, chunk_path=chunked_data_path)