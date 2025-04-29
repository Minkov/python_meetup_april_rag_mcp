from typing import List
from pydantic import Field

from app.schemas.base import AiResponseBaseModel


class InformationNeed(AiResponseBaseModel):
    """Information need identified during reasoning."""
    question: str = Field(description="A specific follow-up question to retrieve information for")
    importance: int = Field(description="Importance score from 1-10", ge=1, le=10)
    context: str = Field(description="Context explaining why this information is needed")


class ResponseWithNeeds(AiResponseBaseModel):
    """Response with identified information needs."""
    response: str = Field(description="The response to the user query based on available information")
    information_needs: List[InformationNeed] = Field(
        description="Identified information needs for follow-up retrieval",
        default=[]
    )
    is_complete: bool = Field(
        description="Whether the response is complete or needs more information",
        default=True
    ) 