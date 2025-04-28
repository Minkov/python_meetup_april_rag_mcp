from typing import List
from app.schemas.retriever import InformationType, RetrieverQueryAnalysis

class QueryComplexityService:
    def __init__(self):
        """Initialize the QueryComplexityService"""
        pass

    def estimate_query_complexity(self, query: str, query_analysis: RetrieverQueryAnalysis) -> float:
        """Estimate the complexity of a query to determine appropriate hierarchy level

        Args:
            query: Search query
            query_analysis: The query analysis object

        Returns:
            Complexity score between 0-1
        """
        complexity = 0.5

        complexity += min(len(query) / 200, 0.3)

        complex_info_types = [
            InformationType.TEAM_STRUCTURE,
            InformationType.PROJECT_DETAILS,
            InformationType.MEETING_NOTES,
        ]
        simple_info_types = [
            InformationType.SPECIFIC_SKILLS,
            InformationType.METRICS,
            InformationType.QUALIFICATIONS
        ]

        for info_type in query_analysis.information_type:
            if info_type in complex_info_types:
                complexity += 0.1
            elif info_type in simple_info_types:
                complexity -= 0.1

        # Adjust based on intent
        intent = query_analysis.intent.lower()
        if any(word in intent for word in ["compare", "analyze", "summarize"]):
            complexity += 0.2
        elif any(word in intent for word in ["find", "locate", "identify"]):
            complexity -= 0.1

        # Ensure within range
        return min(max(complexity, 0.0), 1.0)

    def select_hierarchy_level(self, query_complexity: float) -> str:
        """Select the appropriate hierarchy level based on query complexity

        Args:
            query_complexity: Complexity score between 0-1

        Returns:
            Hierarchy level to use ("base", "section", or "document")
        """
        if query_complexity < 0.4:
            return "base"  # Specific factual queries
        elif query_complexity < 0.7:
            return "section"  # Topic-centered queries
        else:
            return "document"  # Complex, multi-topic queries 