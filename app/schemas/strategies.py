from enum import Enum


class ReasoningStrategy(str, Enum):
    """Enum for different reasoning strategies."""
    DIRECT = "direct"
    COT = "chain_of_thought"
    MULTI_STEP = "multi_step"


class ContextStrategy(str, Enum):
    """Enum for different context handling strategies."""
    FULL = "full"
    RELEVANCE_RANKED = "relevance_ranked"
    TRUNCATED = "truncated"
    COMPRESSED = "compressed"
    MULTI_QUERY = "multi_query"