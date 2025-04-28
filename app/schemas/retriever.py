from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.base import AiResponseBaseModel


class DocumentType(Enum):
    """Enum for the type of the document."""

    JOB_OPENING = "job_opening"
    MEETING_NOTES = "meeting_notes"
    RESUME = "resume"
    TEAM_STRUCTURE = "team_structure"

class Entity(Enum):
    """Enum for the entity of the query."""

    PERSON = "person"
    TEAM = "team"
    JOB_OPENING = "job_opening"
    MEETING_NOTES = "meeting_notes"

class InformationType(Enum):
    """Enum for the information type of the query."""

    JOB_OPENING = "job_opening"
    TEAM_STRUCTURE = "team_structure"
    RESUME = "resume"
    MEETING_NOTES = "meeting_notes"
    PROJECT_DETAILS = "project_details"
    SPECIFIC_SKILLS = "specific_skills"
    METRICS = "metrics"
    QUALIFICATIONS = "qualifications"


class RetrieverQueryAnalysis(AiResponseBaseModel):
    """Schema for analyzing HR queries to understand information needs and intent."""
    
    information_type: List[InformationType] = Field(
        description="Types of information being sought",
        examples=[x.value for x in InformationType]
    )
    
    entities: List[Entity] = Field(
        description="Specific entities mentioned in the query",
        examples=[x.value for x in Entity]
    )
    
    intent: str = Field(
        description="The primary intent behind the query (e.g., matching candidates to jobs, finding information about teams)"
    )
    
    implicit_needs: Optional[List[str]] = Field(
        default=[],
        description="Any implicit information needs not directly mentioned in the query"
    )

    people_identified: Optional[List[str]] = Field(
        default=[],
        description="Any people identified in the query"
    )

    teams_identified: Optional[List[str]] = Field(
        default=[],
        description="Any teams identified in the query"
    )

    jobs_identified: Optional[List[str]] = Field(
        default=[],
        description="Any jobs, job descriptions, job openings or job names identified in the query"
    )


class RetrieverOptimizedQuery(BaseModel):
    """Schema for the optimized query response for vector database retrieval."""
    
    optimized_query: str = Field(
        description="The optimized query, enhanced for semantic retrieval with key terms and entities"
    )


class RetrieverOptimizedQueries(AiResponseBaseModel):
    """Schema for the optimized query response for vector database retrieval."""
    
    optimized_queries: List[RetrieverOptimizedQuery] = Field(
        description="Multiple optimized queries that will be more effective for semantic retrieval"
    )

class RetrieverScoredDocument(AiResponseBaseModel):
    """Schema for the scored document response."""
    
    doc_id: str = Field(
        description="The doc_id of the document"
    )

    relevance_score: float = Field(
        description="The relevance score of the document"
    )

class RetrieverDocScoresResult(AiResponseBaseModel):
    """Schema for the filtered results based on query analysis."""
    
    documents: List[RetrieverScoredDocument] = Field(
        description="List of the documents that with their relevance scores"
    )
    
    filtering_criteria: List[str] = Field(
        description="Criteria used to filter and rank the documents"
    )

class ResultAnswer(Enum):
    """Enum for the answer to the query."""
    
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"

class InformationGapType(Enum):
    """Enum for the information type of the query."""

    JOB_OPENING = "job_opening"
    TEAM_STRUCTURE = "team_structure"
    RESUME = "resume"
    MEETING_NOTES = "meeting_notes"

class InformationGap(AiResponseBaseModel):
    """Schema for an information gap."""
    
    information_gap: str = Field(
        description="The information gap"
    )

    information_type: InformationGapType = Field(
        description="The type of information that is missing",
        examples=[x.value for x in InformationGapType]
    )

class RetrieverResultsAnalysis(AiResponseBaseModel):
    """Schema for analyzing retrieved results to determine if more information is needed."""
    
    status: ResultAnswer = Field(
        description="Whether the information is COMPLETE or INCOMPLETE",
        examples=[x.value for x in ResultAnswer]
    )
    
    explanation: str = Field(
        description="Brief explanation of why the information is complete or what is missing"
    )
    
    information_gaps: Optional[List[InformationGap]] = Field(
        default=[],
        description="List of specific information gaps",
    )
