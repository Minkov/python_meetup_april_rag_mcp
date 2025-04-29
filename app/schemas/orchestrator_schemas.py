from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.schemas.document_results import DocumentResults

@dataclass
class RetrievalRequest:
    """A request from the MCP to retrieve additional information."""
    query: str
    context: Optional[str] = None
    filter_criteria: Optional[Dict[str, Any]] = None
    max_results: int = 5

@dataclass
class OrchestratorResult:
    """Result from the orchestrator including response and debug info."""
    response: str
    retrieved_docs: List[DocumentResults]
    follow_up_queries: List[str]
    total_retrievals: int
    query_analysis: Dict[str, Any] 