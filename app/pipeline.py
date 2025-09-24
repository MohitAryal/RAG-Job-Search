from app.config import settings
from pathlib import Path
from app.helpers.preprocessing import preprocess_dataset

# 1. Preprocess the dataset and store it as json
processed_file = Path(settings.processed_data_dir) / settings.processed_file_name
if not processed_file.exists():
    data_path = Path(settings.raw_data_dir) / settings.file_name
    preprocess_dataset(raw_data_path = data_path, processed_data_path=processed_file)

# 2. Chunk the data to store to the vector db