from typing import List, Dict, Any, Optional, Union
import logging
import re

from app.ai_service import AiServiceCapability, AiServicePrompt, get_ai_service
from app.configs import FAST_GEMINI_MODEL
from app.schemas.document_results import DocumentResults
from app.schemas.strategies import ReasoningStrategy, ContextStrategy
from app.schemas.response_schemas import InformationNeed, ResponseWithNeeds
from app.services.reasoning_strategy_service import ReasoningStrategyService
from app.services.context_handling_service import ContextHandlingService
from app.services.response_generation_service import ResponseGenerationService
from app.services.context_relevance_service import ContextRelevanceService

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelContextProtocol:
    """Enhanced ModelContextProtocol with information need identification capabilities."""

    def __init__(
        self,
        model: str = FAST_GEMINI_MODEL,
        max_tokens: int = 1000,
        temperature: float = 0.3,
        relevance_threshold: float = 4.0,
        enable_logging: bool = True
    ):
        """Initialize the Enhanced ModelContextProtocol.

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
        self.ai_service = get_ai_service(AiServiceCapability.THINKING)
        self.reasoning_strategy_service = ReasoningStrategyService()
        self.context_handling_service = ContextHandlingService(
            model, max_tokens, temperature)
        self.response_generation_service = ResponseGenerationService(
            model, max_tokens, temperature)
        self.context_relevance_service = ContextRelevanceService(model)

    def generate_response(
        self,
        query: str,
        context: Union[str, List[DocumentResults]],
        reasoning_strategy: Optional[ReasoningStrategy] = None,
        context_strategy: Optional[ContextStrategy] = None,
        previous_context: str = None,
        identify_information_needs: bool = True
    ) -> ResponseWithNeeds:
        """Generate a response to a query using the provided context.

        Args:
            query: User query
            context: Context information (string or DocumentResults)
            reasoning_strategy: Reasoning strategy to use (auto-determined if None)
            context_strategy: Context handling strategy to use (auto-determined if None)
            previous_context: Previous conversation context
            identify_information_needs: Whether to identify information needs

        Returns:
            Generated response with information needs if requested
        """
        if previous_context:
            query_with_previous_context = f"{query}\n\nPrevious Context:\n{previous_context}"
        else:
            query_with_previous_context = query

        # Format the context if it's a list of DocumentResults
        if isinstance(context, list):
            formatted_context = self.context_handling_service.format_context(
                context)
        else:
            formatted_context = context

        # Determine reasoning strategy if not provided
        if reasoning_strategy is None:
            reasoning_strategy = self.reasoning_strategy_service.determine_strategy(
                query)
            logger.info(
                f"Dynamically selected reasoning strategy: {reasoning_strategy}")
        else:
            logger.info(
                f"Using provided reasoning strategy: {reasoning_strategy}")

        # Determine context strategy if not provided
        if context_strategy is None:
            context_strategy = self.context_handling_service.determine_strategy(
                query, formatted_context)
            logger.info(
                f"Dynamically selected context strategy: {context_strategy}")
        else:
            logger.info(
                f"Using provided context strategy: {context_strategy}")

        # Apply the selected context strategy
        processed_context = self.context_handling_service.apply_strategy(
            formatted_context, context_strategy, query)

        if self.enable_logging:
            context_tokens = self.context_handling_service._token_estimator(
                processed_context)
            logger.info(f"Context size (estimated): {context_tokens} tokens")

        # If we don't need to identify information needs, use the original generation methods
        if not identify_information_needs:
            if reasoning_strategy == ReasoningStrategy.DIRECT:
                return self.response_generation_service.generate_direct_response(
                    query_with_previous_context, processed_context)
            elif reasoning_strategy == ReasoningStrategy.COT:
                return self.response_generation_service.generate_chain_of_thought_response(
                    query_with_previous_context, processed_context)
            elif reasoning_strategy == ReasoningStrategy.MULTI_STEP:
                return self.response_generation_service.generate_multi_step_response(
                    query_with_previous_context, processed_context)
            else:
                logger.warning(
                    f"Unknown reasoning strategy: {reasoning_strategy}. Using direct generation instead.")
                return self.response_generation_service.generate_direct_response(
                    query_with_previous_context, processed_context)
        
        return self._generate_response_with_needs(
            query_with_previous_context, 
            processed_context, 
            reasoning_strategy
        )

    def _generate_response_with_needs(
        self,
        query: str,
        context: str,
        reasoning_strategy: ReasoningStrategy
    ) -> str:
        """Generate a response that identifies information needs.

        Args:
            query: User query with previous context
            context: Processed context
            reasoning_strategy: Reasoning strategy to use

        Returns:
            Generated response with identified information needs
        """
        # System prompt for response with information needs
        system_prompt = """
You are an HR Intelligence Assistant, specialized in answering HR queries using provided context.

INSTRUCTIONS:
1. Answer the query using ONLY information in the provided context
2. If the context doesn't contain all the necessary information to fully answer the query:
   - Provide the best possible answer with the available information
   - Identify specific pieces of information that are missing
   - Formulate clear follow-up questions that would help complete the answer
3. For each follow-up question, explain why this information is needed
4. Indicate how important each missing piece of information is (on a scale from 1-10)
5. If the answer is complete and no additional information is needed, indicate that

Remember:
- Don't make up information not found in the context
- Be specific about what information is missing
- Frame follow-up questions to focus on specific missing details rather than general topics
    - Keep them brief and targeted to the original query
- Only ask for information that is relevant to answering the original query
"""

        # Formulate the user prompt
        user_prompt = f"""
QUERY: {query}

CONTEXT:
{context}

Based on this context, please provide:
1. Your answer to the query based on available information
2. Any specific follow-up questions needed to provide a complete answer
"""

        # Prepare the prompt object
        prompt = AiServicePrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        try:
            result = self.ai_service.generate_response(
                response_model=ResponseWithNeeds,
                prompt=prompt,
            )
            return result
                
        except Exception as e:
            logger.error(f"Error generating response with needs: {e}")
            # Fall back to the standard response generation
            if reasoning_strategy == ReasoningStrategy.DIRECT:
                return self.response_generation_service.generate_direct_response(
                    query, context)
            elif reasoning_strategy == ReasoningStrategy.COT:
                return self.response_generation_service.generate_chain_of_thought_response(
                    query, context)
            elif reasoning_strategy == ReasoningStrategy.MULTI_STEP:
                return self.response_generation_service.generate_multi_step_response(
                    query, context)
            else:
                return self.response_generation_service.generate_direct_response(
                    query, context)

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