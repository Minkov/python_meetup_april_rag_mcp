from typing import List
import logging
from openai import OpenAI

from app.ai_service import AiServiceCapability, AiServicePrompt, get_ai_service
from app.configs import FAST_GEMINI_MODEL
from app.schemas.base import AiResponseBaseModel
from app.schemas.document_results import DocumentResults
from app.schemas.strategies import ContextStrategy

logger = logging.getLogger(__name__)


class CompressedContext(AiResponseBaseModel):
    compressed_context: str


class ContextHandlingService:
    def __init__(
        self,
        model: str = FAST_GEMINI_MODEL,
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ):
        self.ai_service = get_ai_service(AiServiceCapability.FAST)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._token_estimator = lambda text: len(text) // 4

    def determine_strategy(self, query: str, context: str) -> ContextStrategy:
        """Determine the best context strategy based on query and context.

        Args:
            query: User query
            context: Context information

        Returns:
            Recommended context strategy
        """
        # Estimate context size
        context_tokens = self._token_estimator(context)

        # Simple heuristic based on context size
        if context_tokens < 4000:
            return ContextStrategy.FULL
        elif context_tokens < 7000:
            return ContextStrategy.RELEVANCE_RANKED
        else:
            return ContextStrategy.COMPRESSED

    def format_context(self, documents: List[DocumentResults]) -> str:
        """Format a list of DocumentResults into a context string."""
        context_parts = []

        for i, doc in enumerate(documents, start=1):
            # Add document header
            doc_type = doc.metadata.get("document_type", "document")
            context_parts.append(f"Document {i} ({doc_type}):")

            # Add document ID for reference
            context_parts.append(f"Document ID: {doc.doc_id}")

            # Add metadata if available
            if doc.metadata:
                # Filter out less relevant metadata
                relevant_metadata = {k: v for k, v in doc.metadata.items()
                                     if k not in ['hierarchy_level', 'parent_id', 'base_chunks',
                                                  'section_chunks', 'hierarchy_index', 'token_count',
                                                  'next_chunk_id', 'prev_chunk_id']}

                if relevant_metadata:
                    context_parts.append("Metadata:")
                    for key, value in relevant_metadata.items():
                        context_parts.append(f"- {key}: {value}")

            # Add document content
            context_parts.append("Content:")
            context_parts.append(doc.text)
            context_parts.append("-" * 40)  # Separator

        return "\n".join(context_parts)

    def apply_strategy(
        self,
        context: str,
        strategy: ContextStrategy,
        query: str
    ) -> str:
        """Apply a context handling strategy to the context."""
        if strategy == ContextStrategy.FULL:
            return context

        elif strategy == ContextStrategy.TRUNCATED:
            # Simple truncation to fit within estimated token limits
            max_context_tokens = 8000

            # Estimate current tokens
            estimated_tokens = self._token_estimator(context)

            if estimated_tokens <= max_context_tokens:
                return context

            # Truncate to fit
            truncation_ratio = max_context_tokens / estimated_tokens
            truncated_length = int(len(context) * truncation_ratio)
            truncated_context = context[:truncated_length]

            # Add note about truncation
            return truncated_context + "\n\n[Note: Context was truncated due to length limits.]"

        elif strategy == ContextStrategy.RELEVANCE_RANKED:
            # For this strategy, we assume context is already in relevance order
            # This is typically handled by the retriever, so we just pass through
            return context

        elif strategy == ContextStrategy.COMPRESSED:
            # Use the model to compress the context while preserving important information
            return self._compress_context(context, query)

        else:
            logger.warning(
                f"Unknown context strategy: {strategy}. Using full context instead.")
            return context

    def _compress_context(self, context: str, query: str) -> str:
        """Compress the context while preserving information relevant to the query."""
        system_prompt = """
You are an AI assistant specialized in condensing and summarizing information while preserving all facts that might be relevant to answering a specific query.

Your task is to:
1. Analyze the provided context documents and the user query
2. Identify key information in the documents that might help answer the query
3. Create a compressed version of the context that:
   - Preserves all factual information relevant to the query
   - Removes redundant or irrelevant details
   - Maintains document organization and source references
   - Reduces the overall token count significantly

Keep document boundaries and metadata intact, but compress the content of each document.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-16k",  # Using a larger context model for compression
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
Query: {query}

Original context:
{context}

Please create a compressed version of this context that preserves all information that might help answer the query.
                    """}
                ],
                temperature=0.3,
                max_tokens=4000
            )

            prompt = AiServicePrompt(
                system_prompt=system_prompt,
                user_prompt=f"""
Query: {query}

Original context:
{context}

Please create a compressed version of this context that preserves all information that might help answer the query.
"""
            )

            compressed_context = self.ai_service.generate_response(
                prompt=prompt,
                response_model=CompressedContext
            )

            return compressed_context + "\n\n[Note: Context was compressed to preserve relevant information while reducing length.]"

        except Exception as e:
            logger.error(f"Error compressing context: {e}")
            # Fall back to truncation
            return self.apply_strategy(context, ContextStrategy.TRUNCATED, query)

    def set_token_estimator(self, estimator_func: callable):
        """Set a custom token estimator function.

        Args:
            estimator_func: Function that takes a string and returns estimated token count
        """
        self._token_estimator = estimator_func
