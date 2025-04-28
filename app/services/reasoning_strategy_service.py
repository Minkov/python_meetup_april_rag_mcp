from typing import Type
from pydantic import Field

from app.ai_service import AiServiceCapability, AiServicePrompt, get_ai_service
from app.configs import FAST_GEMINI_MODEL
from app.schemas.base import AiResponseBaseModel
from app.schemas.strategies import ReasoningStrategy


class ReasoningStrategyResponse(AiResponseBaseModel):
    strategy: ReasoningStrategy = Field(
        description="The best reasoning strategy for the query",
        values=[x.value for x in ReasoningStrategy],
    )


class ReasoningStrategyService:
    def __init__(self):
        self.ai_service = get_ai_service(AiServiceCapability.FAST)

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
1. DIRECT - Simple, straightforward questions that can be answered directly from the context. 
   Examples: "What is the position of Peter in the company?", "What are the responsibilities of the Senior Data Engineer?"

2. COT (Chain-of-Thought) - Questions requiring step-by-step reasoning but solvable in one pass.
   Examples: "Analyze the resume of George Michaelson, compare it to the job description of the Senior Data Engineer and provide a match score.",
             "Explain why John Smith is a good fit for the Junior Web Engineer position."

3. MULTI_STEP - Complex questions that should be broken down into sub-questions and answered sequentially.
   Examples: "Evaluate our current recruitment process for engineering roles and suggest improvements based on industry best practices.",
             "Analyze the career progression of our data scientists over the past 3 years and recommend changes to our promotion criteria."


When choosing a strategy, prefer simpler approaches when possible:
- First consider if DIRECT is sufficient
- If not, consider if COT would work
- Only choose MULTI_STEP for the most complex questions that cannot be reasonably addressed with the other strategies


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
