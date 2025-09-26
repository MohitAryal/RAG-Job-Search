from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    
    # Vector Database Configuration
    vector_db_url: str = Field(default="http://localhost:6333", alias="VECTOR_DB_URL")
    vector_db_api_key: Optional[str] = Field(default=None, alias="VECTOR_DB_API_KEY")
    vector_db_collection_name: str = Field(default="job_documents", alias="VECTOR_DB_COLLECTION_NAME")
    vector_size: int = Field(default=384, alias="VECTOR_SIZE")
    
    # LLM Configuration
    llm_model: str = Field(default="llama3-70b-8192", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1000, alias="LLM_MAX_TOKENS")
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")
    
    # Embedding Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(default=100, alias="EMBEDDING_BATCH_SIZE")
    
    # Search Configuration
    default_top_k: int = Field(default=20, alias="DEFAULT_TOP_K")
    max_top_k: int = Field(default=50, alias="MAX_TOP_K")
    similarity_threshold: float = Field(default=0.7, alias="SIMILARITY_THRESHOLD")
    
    # Reranking
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL")
    reranker_top_k: int = Field(default=5, alias="RERANKER_TOP_K")
    
    # Keyword Search
    bm25_k1: float = Field(default=1.2, alias="BM25_K1")
    bm25_b: float = Field(default=0.75, alias="BM25_B")
    
    # Hybrid Search
    k: float = Field(default=60, alias="K")

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(default=10, alias="RATE_LIMIT_BURST")
    
    # Data Directories
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    raw_data_dir: str = Field(default="./data/raw", alias="RAW_DATA_DIR")
    processed_data_dir: str = Field(default="./data/processed", alias="PROCESSED_DATA_DIR")
    chunked_data_dir: str = Field(default="./data/chunks", alias="CHUNKED_DATA_DIR")
    embeddings_data_dir: str = Field(default="./data/embeddings", alias="EMBEDDINGS_DATA_DIR")
    keyword_retriever_dir: str = Field(default="./data/keyword_retriever", alias="KEYWORD_RETRIEVER_DIR")

    # File name
    file_name: str = Field(default='LF Jobs.xlsx', alias='FILE_NAME')
    processed_file_name: str = Field(default='processed.json', alias='PROCESSED_FILE_NAME')
    chunked_file_name: str = Field(default='chunked.json', alias='CHUNKED_FILE_NAME')
    embeddings_file_name: str = Field(default='embeddings.npy', alias='EMBEDDINGS_FILE_NAME')
    keyword_retriever_file: str = Field(default='keyword_retriever.pkl', alias='KEYWORD_RETRIVER_FILE')
    
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()