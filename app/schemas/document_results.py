from dataclasses import dataclass
from typing import Dict

@dataclass
class DocumentResults:
    """Schema for the document results response."""
    
    doc_id: str
    similarity: float
    text: str
    metadata: Dict[str, str]
