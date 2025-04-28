from enum import Enum


class ReasoningStrategy(str, Enum):
    """Enum for different reasoning strategies."""
    DIRECT = "direct"  # Directly use context to answer
    COT = "chain_of_thought"  # Use chain-of-thought reasoning
    ROT = "retrieval_of_thought"  # Retrieve, then reason step by step
    MULTI_STEP = "multi_step"  # Break query into sub-questions


class ContextStrategy(str, Enum):
    """Enum for different context handling strategies."""
    FULL = "full"  # Use the full context provided
    RELEVANCE_RANKED = "relevance_ranked"  # Order context by relevance
    TRUNCATED = "truncated"  # Truncate context to fit token limit
    COMPRESSED = "compressed"  # Compress context before sending to model
    MULTI_QUERY = "multi_query"  # Break into multiple queries with different contexts 