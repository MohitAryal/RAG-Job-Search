from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for job search queries."""
    
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=200,
        description="Search query for job listings",
        example="senior data scientist machine learning"
    )
    
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of results to return (1–10)",
        example=3
    )
    
    include_full_jd: bool = Field(
        default=False,
        description="If True, include full job descriptions in results"
    )

    @validator('query')
    def validate_query(cls, v):
        """Ensure query is not empty/whitespace and has alphanumeric content."""
        if not v.strip():
            raise ValueError('Query cannot be empty or just whitespace')
        if not any(c.isalnum() for c in v):
            raise ValueError('Query must contain at least one alphanumeric character')
        return v.strip()


class JobLocation(BaseModel):
    """Model for job location information."""
    
    cities: List[str] = Field(default=[], description="List of cities")
    states: List[str] = Field(default=[], description="List of states")
    countries: List[str] = Field(default=[], description="List of countries")
    is_remote: bool = Field(default=False, description="Whether the job is remote")


class SearchResult(BaseModel):
    """Model for individual search result."""
    
    rank: int = Field(..., description="Ranking position of this result (1-based)")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    category: str = Field(..., description="Job category")
    level: str = Field(..., description="Job level/seniority")
    location: JobLocation = Field(..., description="Job location information")
    tags: List[str] = Field(default=[], description="Job tags")
    
    content_snippet: str = Field(
        ..., 
        description="Relevant content snippet from job description",
        max_length=150
    )
    
    full_description: Optional[str] = Field(
        None,
        description="Full job description (included only if requested)"
    )
    
    explanation: Optional[str] = Field(
        None,
        description="LLM-generated explanation for why this job is relevant"
    )


class QueryResponse(BaseModel):
    """Response model for job search queries."""    
    
    answer: Optional[str] = Field(
        None,
        description="LLM-generated overall summary of the results"
    )
    
    results: List[SearchResult] = Field(
        ..., 
        description="List of ranked search results"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the search"
    )