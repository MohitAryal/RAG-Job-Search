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


class JobLocation(BaseModel):
    """Model for job location information."""
    
    cities: List[str] = Field(default=[], description="List of cities")
    states: List[str] = Field(default=[], description="List of states")
    countries: List[str] = Field(default=[], description="List of countries")
    is_remote: bool = Field(default=False, description="Whether the job is remote")


class SearchResult(BaseModel):
    """Model for individual search result."""
    
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    category: str = Field(..., description="Job category")
    level: str = Field(..., description="Job level/seniority")
    location: JobLocation = Field(..., description="Job location information")
    tags: List[str] = Field(default=[], description="Job tags")
    
    description: str = Field(
        ...,
        description="Full job description"
    )
    
    explanation: str = Field(
        ...,
        description="Explanation for why this job is relevant"
    )


class QueryResponse(BaseModel):
    """Response model for job search queries."""    
    
    results: Optional[List[SearchResult]] = Field(None, description="List of relevant job matches")
    message: Optional[str] = Field(None, description="Message explaining why no jobs matched the query")
   