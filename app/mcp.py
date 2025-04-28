from typing import List, Dict, Any, Optional, Union, Type
import logging

from app.configs import FAST_GEMINI_MODEL
from app.schemas.base import AiResponseBaseModel
from app.schemas.document_results import DocumentResults
from app.schemas.strategies import ReasoningStrategy, ContextStrategy
from app.services.reasoning_strategy_service import ReasoningStrategyService
from app.services.context_handling_service import ContextHandlingService
from app.services.response_generation_service import ResponseGenerationService
from app.services.context_relevance_service import ContextRelevanceService

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelContextProtocol:
    """Enhanced ModelContextProtocol with advanced context handling and reasoning strategies."""

    def __init__(
        self,
        model: str = FAST_GEMINI_MODEL,
        max_tokens: int = 1000,
        temperature: float = 0.3,
        relevance_threshold: float = 4.0,
        enable_logging: bool = True
    ):
        """Initialize the ModelContextProtocol.

        Args:
            model: Model to use for generation
            max_tokens: Maximum tokens in the response
            temperature: Temperature for generation
            relevance_threshold: Threshold for considering context relevant
            enable_logging: Whether to enable logging
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.relevance_threshold = relevance_threshold
        self.enable_logging = enable_logging

        # Initialize services
        self.reasoning_strategy_service = ReasoningStrategyService()
        self.context_handling_service = ContextHandlingService(
            model, max_tokens, temperature)
        self.response_generation_service = ResponseGenerationService(
            model, max_tokens, temperature)
        self.context_relevance_service = ContextRelevanceService(
            model, max_tokens, temperature)

    def generate_response(
        self,
        query: str,
        context: Union[str, List[DocumentResults]],
        reasoning_strategy: Optional[ReasoningStrategy] = None,
        context_strategy: Optional[ContextStrategy] = None,
        structured_output_format: Optional[Type[AiResponseBaseModel]] = None
    ) -> Union[str, Dict[str, Any], AiResponseBaseModel]:
        """Generate a response to a query using the provided context.

        Args:
            query: User query
            context: Context information (string or DocumentResults)
            reasoning_strategy: Reasoning strategy to use (auto-determined if None)
            context_strategy: Context handling strategy to use (auto-determined if None)
            structured_output_format: Optional pydantic model class for structured output

        Returns:
            Generated response as string, dict, or structured object
        """

        if isinstance(context, list):
            formatted_context = self.context_handling_service.format_context(
                context)
        else:
            formatted_context = context

        if reasoning_strategy is None:
            reasoning_strategy = self.reasoning_strategy_service.determine_strategy(
                query)
            logger.info(
                f"Dynamically selected reasoning strategy: {reasoning_strategy}")
        else:
            logger.info(
                f"Using provided reasoning strategy: {reasoning_strategy}")

        if context_strategy is None:
            context_strategy = self.context_handling_service.determine_strategy(
                query, formatted_context)
            logger.info(
                f"Dynamically selected context strategy: {context_strategy}")
        else:
            logger.info(
                f"Using provided context strategy: {context_strategy}")

        if isinstance(context, list):
            context = self.context_handling_service.format_context(context)

        processed_context = self.context_handling_service.apply_strategy(
            context, context_strategy, query)

        if self.enable_logging:
            context_tokens = self.context_handling_service._token_estimator(
                processed_context)
            logger.info(f"Context size (estimated): {context_tokens} tokens")

        if reasoning_strategy == ReasoningStrategy.DIRECT:
            return self.response_generation_service.generate_direct_response(
                query, processed_context)
        elif reasoning_strategy == ReasoningStrategy.COT:
            return self.response_generation_service.generate_chain_of_thought_response(
                query, processed_context)
        elif reasoning_strategy == ReasoningStrategy.MULTI_STEP:
            return self.response_generation_service.generate_multi_step_response(
                query, processed_context)
        else:
            logger.warning(
                f"Unknown reasoning strategy: {reasoning_strategy}. Using direct generation instead.")
            return self.response_generation_service.generate_direct_response(
                query, processed_context)

    def assess_context_relevance(self, query: str, context: Union[str, List[DocumentResults]]) -> Dict[str, Any]:
        """Assess if the context is relevant and sufficient for answering the query.

        Args:
            query: User query
            context: Context information

        Returns:
            Dictionary with relevance assessment
        """
        return self.context_relevance_service.assess_relevance(
            query, context, self.context_handling_service.format_context)
