from typing import List
from pydantic import Field

from app.schemas.base import AiResponseBaseModel


class RelevanceAssessment(AiResponseBaseModel):
    """Schema for context relevance assessment."""
    relevant: bool = Field(description="Whether the context is relevant to the query")
    relevance_score: float = Field(description="Relevance score from 0-10")
    explanation: str = Field(description="Explanation of the relevance assessment")
    missing_information: List[str] = Field(description="List of missing information needed to answer the query") 