import pandas as pd 
from app.config import settings
from pathlib import Path

data_path = Path(settings.raw_data_dir) / settings.file_name
df = pd.read_excel(data_path)

print(df.head())