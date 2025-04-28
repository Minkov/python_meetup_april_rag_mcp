from typing import Type
from pydantic import Field

from app.ai_service import AiServicePrompt, GenimiAiService
from app.configs import FAST_GEMINI_MODEL
from app.schemas.base import AiResponseBaseModel
from app.schemas.strategies import ReasoningStrategy


class ReasoningStrategyResponse(AiResponseBaseModel):
    strategy: ReasoningStrategy = Field(
        description="The best reasoning strategy for the query",
        values=[x.value for x in ReasoningStrategy],
    )


class ReasoningStrategyService:
    def __init__(self, model: str = FAST_GEMINI_MODEL):
        self.ai_service = GenimiAiService(model)

    def determine_strategy(self, query: str) -> ReasoningStrategy:
        """Determine the best reasoning strategy for a query.

        Args:
            query: User query

        Returns:
            Recommended reasoning strategy
        """
        prompt = f"""
You are an AI assistant that analyzes queries to determine the best reasoning strategy.

For a given query, determine the most appropriate reasoning strategy from the following options:
1. DIRECT - Simple, straightforward questions that can be answered directly from the context
2. COT (Chain-of-Thought) - Complex questions requiring step-by-step reasoning
3. MULTI_STEP - Complex questions that should be broken down into sub-questions

Return ONLY one of these three values: {", ".join([x.value for x in ReasoningStrategy])}
"""

        user_prompt = f"Analyze this query and determine the best reasoning strategy: {query}"

        prompt_obj = AiServicePrompt(
            system_prompt=prompt,
            user_prompt=user_prompt
        )

        try:
            response = self.ai_service.generate_response(
                response_model=ReasoningStrategyResponse,
                prompt=prompt_obj,
            )

            return response.strategy

        except Exception as e:
            # Default to direct if there's an error
            return ReasoningStrategy.DIRECT
