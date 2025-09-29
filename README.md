# Job Search RAG System

An intelligent job search system powered by Retrieval-Augmented Generation (RAG) that combines keyword search, semantic search, and LLM-based explanation to deliver accurate and interpretable job recommendations.

## Features

- **Hybrid Search**: Combines BM25 keyword search with semantic vector search for optimal precision and recall
- **Smart Reranking**: Cross-encoder model ensures the most relevant results appear first
- **AI-Powered Explanations**: LLM provides human-readable justifications for each job match
- **Scalable Architecture**: Cloud-based vector storage with Qdrant
- **Semantic Understanding**: Chunks job descriptions into meaningful sections for better context matching

## Architecture

The system consists of two main pipelines:

### Ingestion Pipeline
1. **Data Loading**: Ingests job postings from Excel file
2. **Cleaning & Normalization**: Standardizes job metadata
3. **Intelligent Chunking**: Splits descriptions into semantic sections (Responsibilities, Requirements, Benefits)
4. **BM25 Indexing**: Creates keyword search index using rank-bm25
5. **Embedding Generation**: Generates dense embeddings with all-MiniLM-L6-v2
6. **Vector Storage**: Stores embeddings in Qdrant Cloud

### Query Pipeline
1. **Hybrid Retrieval**: Retrieves results from both BM25 and Qdrant
2. **Reciprocal Rank Fusion**: Intelligently merges rankings from both sources
3. **Cross-Encoder Reranking**: Rescores results for semantic alignment
4. **LLM Verification**: Generates explanations via Groq API
5. **Response Generation**: Returns job matches with interpretable justifications

## Prerequisites

- Python 3.8+
- Qdrant Cloud account
- Groq API key
- Job postings data in Excel format

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/MohitAryal/RAG-Job-Search.git
cd RAG-job-Search
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the root directory:

```env
# Vector Database Configuration - Qdrant
VECTOR_DB_URL=your_qdrant_cluster_url
VECTOR_DB_API_KEY=your_qdrant_api_key
VECTOR_DB_COLLECTION_NAME=job_search_collection
VECTOR_SIZE=384

# LLM Configuration - Groq
LLM_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=1000
LLM_API_KEY=your_groq_api_key

# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=32

# Search Configuration
DEFAULT_TOP_K=20

# Reranking
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_K=5

# Keywords Search
BM25_K1=1.5
BM25_B=0.75

# Hybrid Search
K=60

# Data Directories
DATA_DIR=app/data
RAW_DATA_DIR=app/data/raw
PROCESSED_DATA_DIR=app/data/processed
CHUNKED_DATA_DIR=app/data/chunks
EMBEDDINGS_DATA_DIR=app/data/embeddings
KEYWORD_RETRIEVER_DIR=app/data/keyword_retriever

# File Names
FILE_NAME=LF Jobs.xlsx
PROCESSED_FILE_NAME=processed.json
CHUNKED_FILE_NAME=chunked.json
EMBEDDINGS_FILE_NAME=embeddings.npy
KEYWORD_RETRIVER_FILE=keyword_retriver.pkl
```

## Data Preparation

Place your job postings Excel file (`LF Jobs.xlsx`) in the `app/data/raw/` directory.

## Usage

### 1. Run Ingestion Pipeline

**Important**: This must be executed before starting the API server.

```bash
python -m app.ingestion_pipeline
```

This will:
- Process your job postings
- Generate embeddings
- Build BM25 index
- Upload vectors to Qdrant

### 2. Start the API Server

```bash
uvicorn app.main:router --reload
```

The API will be available at `http://localhost:8000`

### 3. Query Jobs

**Endpoint**: `POST /api/query`

**Request Example**:
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "remote senior data scientist"}'
```

**Response Example**:
```
Job ID: J0023
Title: Senior Data Scientist
Location: Remote (USA)
Explanation: This job matches because it is a senior role, remote-friendly, and requires machine learning expertise.

Job ID: J0344
Title: Lead Machine Learning Engineer
Location: Remote (Canada/USA)
Explanation: This role emphasizes machine learning leadership and supports remote work, aligning with the query.
```

## How It Works

1. **User submits a query** (e.g., "remote python developer")
2. **System performs hybrid search** using both keyword matching and semantic similarity
3. **Results are merged** using Reciprocal Rank Fusion
4. **Cross-encoder reranks** the top candidates for better accuracy
5. **LLM validates and explains** why each job matches the query
6. **User receives** ranked jobs with clear, interpretable explanations

## Technology Stack

- **FastAPI**: Web framework
- **Qdrant**: Vector database
- **LangChain**: LLM orchestration
- **Groq**: LLM inference
- **Sentence Transformers**: Embedding and reranking models
- **rank-bm25**: Keyword search
- **Python**: Core language

## Configuration Tips

- **BM25_K1** (1.2-2.0): Higher values increase the impact of term frequency
- **BM25_B** (0-1): Higher values apply stronger document length normalization
- **K** (RRF constant): Lower values (30-60) favor top-ranked results more
- **LLM_TEMPERATURE**: Lower (0.1-0.3) for consistent results, higher (0.7-1.0) for creative explanations

## Known Limitations

- Ingestion pipeline must be run manually before queries can be served
- No programmatic filters for job attributes (remote, location, seniority)
- LLM adds latency (~1-3 seconds per query)
- Dependent on external API availability (Qdrant, Groq)

## Future Enhancements

- [ ] Structured filtering (location, remote, salary range)
- [ ] Automatic ingestion on startup or scheduled updates
- [ ] Query/result caching to reduce LLM calls
- [ ] User feedback loop for continuous improvement
- [ ] Support for multiple data sources
- [ ] Real-time job posting updates

---

**Note**: Make sure to keep your API keys secure and never commit them to version control. Use `.gitignore` to exclude your `.env` file.