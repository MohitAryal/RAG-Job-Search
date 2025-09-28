from pydantic import BaseModel, Field, validator
from typing import List, Optional
from app.config import settings


class QueryRequest(BaseModel):
    """Request model for job search queries."""
    
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=200,
        description="Search query for job listings",
        example="senior data scientist machine learning"
    )

    @validator('query')
    def validate_query(cls, v):
        """Ensure query is not empty/whitespace"""
        if not v.strip():
            raise ValueError('Query cannot be empty or just whitespace')
        return v.strip()
   