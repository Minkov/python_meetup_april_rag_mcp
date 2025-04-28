from typing import List, Dict, Any, Union
import logging
import json
from openai import OpenAI

from app.ai_service import AiServiceCapability, AiServicePrompt, get_ai_service
from app.configs import FAST_GEMINI_MODEL
from app.schemas.base import AiResponseBaseModel
from app.schemas.document_results import DocumentResults

logger = logging.getLogger(__name__)


class ContextRelevanceService:
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

    def assess_relevance(
        self,
        query: str,
        context: Union[str, List[DocumentResults]],
        format_context_func: callable
    ) -> Dict[str, Any]:
        """Assess if the context is relevant and sufficient for answering the query.

        Args:
            query: User query
            context: Context information
            format_context_func: Function to format context if it's a list of DocumentResults

        Returns:
            Dictionary with relevance assessment
        """
        # Format context if it's a list of DocumentResults
        if isinstance(context, list):
            context = format_context_func(context)

        system_prompt = """
You are an HR Intelligence Assistant specializing in assessing the relevance of context information to a given query.

Your task is to:
1. Analyze the provided context and the user query
2. Determine if the context contains the information needed to answer the query
3. Identify any information gaps or missing details
4. Provide a relevance score from 0-10, where:
   - 0-3: Context is largely irrelevant or missing critical information
   - 4-6: Context contains some relevant information but has significant gaps
   - 7-10: Context contains most or all information needed to answer the query
"""

        user_prompt = f"""
Query: {query}

Context:
{context}

Please assess if this context is relevant and sufficient for answering the query.
"""

        prompt = AiServicePrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        class RelevanceAssessment(AiResponseBaseModel):
            relevant: bool
            relevance_score: float
            explanation: str
            missing_information: List[str]

        try:
            result = self.ai_service.generate_response(
                response_model=RelevanceAssessment,
                prompt=prompt,
            )
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error assessing context relevance: {e}")
            # Use direct OpenAI client as fallback
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )

                result = json.loads(response.choices[0].message.content)
                return result
            except Exception as e2:
                logger.error(f"Fallback assessment also failed: {e2}")
                return {
                    "error": str(e),
                    "relevant": False,
                    "relevance_score": 0,
                    "explanation": "Failed to assess context relevance due to an error",
                    "missing_information": ["Unable to determine"]
                } 