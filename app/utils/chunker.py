import re
import pandas as pd
import json
from app.config import settings

processed_data_path = Path(settings.processed_data_dir) / settings.processed_file_name
chunked_data_path = Path(settings.chunked_data_dir) / settings.chunked_file_name

def generate_chunks(job):
  # Regular expression to capture each section with its name
  sections = re.findall(r'(\*\*.*?\*\*)(.*?)(?=\*\*|$)', job['Job Description'], re.DOTALL)
  metadata = {
          "job_id": job['ID'],
          "title": job['cleaned_title'],
          "company": job['Company Name'],
          "location": job['Job Location'],
          "level": job['Job Level'],
          "category": job['Job Category'],
      }
  # Store sections as a dictionary
  chunks = [{'content': f'{section_name.strip()}\n{section_content.strip()}', 'metadata': metadata, 'chunk_id': f'{metadata["job_id"][2:]}_{i}'} for i, (section_name, section_content) in enumerate(sections)]
  return chunks


def save_chunks_to_json(chunks, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f'Chunks saved')


def chunk_job_descriptions():
    df = pd.read_json(processed_data_path)
    df['chunks'] = df.apply(generate_chunks, axis=1)

    # Flatten all chunks from all rows into a single list    
    all_chunks = [chunk for chunks_list in df['chunks'] for chunk in chunks_list]
    save_chunks_to_json(all_chunks, chunk_path)    