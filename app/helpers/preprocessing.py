from app.config import settings
from pathlib import Path
import pandas as pd
from app.utils.location_cleaner import preprocess_job_location
from app.utils.tags_cleaner import clean_tags
from app.utils.title_cleaner import clean_job_titles
from app.utils.description_cleaner import html_to_markdown


def preprocess_dataset(raw_data_path, processed_data_path):
    """Function to preprocess the dataset and store it as a json file"""
    
    print("Preprocessing the dataset...")
    
    # Load and preprocess data
    df = pd.read_excel(raw_data_path)
    
    # Apply preprocessing
    print('\nCleaning the job location...')
    df['Job Location'] = df['Job Location'].apply(preprocess_job_location)
    
    print('\nCleaning tags...')
    df['Tags'] = df['Tags'].apply(clean_tags)

    print('\nCleaning title...')
    df['cleaned_title'] = df['Job Title'].apply(clean_job_titles)

    print('\nCleaning description...')
    df['Job Description'] = df['Job Description'].apply(html_to_markdown)
    
    # Save as JSON
    df.to_json(processed_data_path, orient='records', indent=2)
    print(f"\nPreprocessing complete. Saved to {processed_data_path}")
    
    return