from typing import List, Dict, Any, Union
import logging
from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ai_service import AiServiceCapability, AiServicePrompt, get_ai_service
from app.configs import FAST_GEMINI_MODEL
from app.schemas.base import AiResponseBaseModel
from app.schemas.strategies import ReasoningStrategy

logger = logging.getLogger(__name__)


class SubAnswerResponse(AiResponseBaseModel):
    answer: str = Field(description="The answer to the sub-question")


class SubQuestionsResponse(AiResponseBaseModel):
    sub_questions: List[str] = Field(description="A list of sub-questions that are needed to answer the main query")

class ChainOfThoughtReasoningStep(AiResponseBaseModel):
    reasoning: str = Field(description="The reasoning for the step")


class ChainOfThoughtReasoningResponse(AiResponseBaseModel):
    reasoning: List[ChainOfThoughtReasoningStep] = Field(description="A list of steps in the chain of thought reasoning process")


class ResultResponse(AiResponseBaseModel):
    response: str = Field(description="The final response to the user query")


class ResponseGenerationService:
    system_prompts = {
        ReasoningStrategy.DIRECT: """
You are an HR Intelligence Assistant, specialized in helping HR professionals find information 
about job descriptions, team structures, candidate resumes, and meeting notes.

When answering questions:
1. Only use information provided in the context.
2. If the context doesn't contain enough information, say so clearly.
3. For job-candidate matching, consider both explicit requirements and implicit skills.
4. When referring to documents, cite the document number as provided in the context.
5. Format your responses in a clear, professional manner suitable for HR specialists.
6. Do not make up information that isn't in the context.
""",

        ReasoningStrategy.COT: """
You are an HR Intelligence Assistant, specialized in helping HR professionals find information 
about job descriptions, team structures, candidate resumes, and meeting notes.

When answering questions:
1. Only use information provided in the context.
2. Break down your reasoning step by step:
   - First identify what information you need to answer the question
   - Then find the relevant pieces from the context
   - Build your answer one step at a time using the relevant information
3. If the context doesn't contain enough information, say so clearly.
4. When referring to documents, cite the document number as provided in the context.
5. Format your responses in a clear, professional manner suitable for HR specialists.
6. Do not make up information that isn't in the context.
""",

        ReasoningStrategy.MULTI_STEP: """
You are an HR Intelligence Assistant, specialized in helping HR professionals find information 
about job descriptions, team structures, candidate resumes, and meeting notes.

Your task is to break down complex questions into smaller, more manageable sub-questions, and then answer each one:
1. Only use information provided in the context.
2. First, identify the main question and any sub-questions needed to fully answer it
3. For each sub-question:
   - Find the relevant information in the context
   - Provide a concise answer with citations
4. Finally, synthesize all sub-answers into a comprehensive response to the main question
5. If the context doesn't contain enough information for any sub-question, say so clearly.
6. When referring to documents, cite the document number as provided in the context.
7. Format your responses in a clear, professional manner suitable for HR specialists.
"""
    }

    def __init__(
        self,
        model: str = FAST_GEMINI_MODEL,
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ):
        self.ai_service = get_ai_service(AiServiceCapability.THINKING)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_direct_response(
        self,
        query: str,
        context: str
    ) -> Union[str, Dict[str, Any], AiResponseBaseModel]:
        """Generate a response using direct generation."""
        system_prompt = self.system_prompts[ReasoningStrategy.DIRECT]

        user_prompt = f"""
Context information:
{context}

User query: {query}

Please answer the user query using only the information provided in the context above.
"""

        prompt = AiServicePrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        result = self.ai_service.generate_response(
            response_model=ResultResponse,
            prompt=prompt,
        )
        return result.response

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_chain_of_thought_response(
        self,
        query: str,
        context: str,
    ) -> Union[str, Dict[str, Any], AiResponseBaseModel]:
        """Generate a response using chain-of-thought reasoning."""
        system_prompt = self.system_prompts[ReasoningStrategy.COT]

        user_prompt = f"""
Context information:
{context}

User query: {query}

Please answer the user query step by step:
1. First analyze what information you need to answer this question
2. Then find the relevant pieces in the context
3. Finally, build your answer systematically
"""

        reasoning_prompt = AiServicePrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        reasoning_response = self.ai_service.generate_response(
            response_model=ChainOfThoughtReasoningResponse,
            prompt=reasoning_prompt,
        )

        reasoning = "\n".join([step.reasoning for step in reasoning_response.reasoning])

        final_system_prompt = f"""
{system_prompt}

You previously provided the following reasoning:
{reasoning}

Now, provide your final answer as a structured output in the format requested.
"""

        final_user_prompt = "Based on your reasoning, provide the final structured answer."

        final_prompt = AiServicePrompt(
            system_prompt=final_system_prompt,
            user_prompt=final_user_prompt
        )

        try:
            result = self.ai_service.generate_response(
                response_model=ResultResponse,
                prompt=final_prompt,
            )
            return result.response
        except Exception as e:
            logger.error(f"Error generating structured response: {e}")
            return {"error": "Failed to parse structured output",
                    "reasoning": reasoning}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_multi_step_response(
        self,
        query: str,
        context: str,
    ) -> Union[str, Dict[str, Any], AiResponseBaseModel]:
        """Generate a response by breaking down the query into sub-questions."""
        # Step 1: Identify sub-questions using AiService
        subq_system_prompt = """
You are an expert at breaking down complex HR queries into simpler sub-questions.
Given a main query, identify 2-4 specific sub-questions that, when answered, would provide a complete response to the main query.
Each sub-question should focus on a single, clear aspect of the information needed.
"""

        subq_user_prompt = f"""
Main query: {query}

Based on this query, what are the specific sub-questions we need to answer?
List each sub-question on a separate line with a number prefix.
"""

        subq_prompt = AiServicePrompt(
            system_prompt=subq_system_prompt,
            user_prompt=subq_user_prompt
        )
        try:
            subq_result = self.ai_service.generate_response(
                response_model=SubQuestionsResponse,
                prompt=subq_prompt,
            )
            sub_questions = subq_result.sub_questions

            if not sub_questions:
                logger.warning(
                    "Failed to parse sub-questions, falling back to direct generation")
                return self.generate_direct_response(query, context)

            # Step 2: Answer each sub-question
            sub_answers = []
            for i, sub_q in enumerate(sub_questions, start=1):
                logger.info(f"Generating answer for sub-question {i}: {sub_q}")

                sub_system_prompt = self.system_prompts[ReasoningStrategy.DIRECT]

                sub_user_prompt = f"""
    Context information:
    {context}

    User sub-query: {sub_q}
    (This is part of the main query: {query})

    Please answer this specific sub-question using only the information in the context.
    Keep your answer concise and directly address this sub-question only.
    """
                sub_prompt = AiServicePrompt(
                    system_prompt=sub_system_prompt,
                    user_prompt=sub_user_prompt
                )

                sub_response = self.ai_service.generate_response(
                    response_model=SubAnswerResponse,
                    prompt=sub_prompt,
                )
                sub_answers.append({
                    "question": sub_q,
                    "answer": sub_response.answer
                })

            # Step 3: Synthesize the answers
            synthesis_content = "Here are the answers to the sub-questions:\n\n"
            for i, item in enumerate(sub_answers, start=1):
                synthesis_content += f"Sub-question {i}: {item['question']}\n"
                synthesis_content += f"Answer: {item['answer']}\n\n"

            synthesis_system_prompt = """
    You are an HR Intelligence Assistant synthesizing answers to sub-questions into a cohesive response.
    Your task is to integrate the answers to multiple sub-questions into a single, comprehensive response to the main query.
    Ensure you maintain all important information from the sub-answers while avoiding repetition.
    Format your response in a clear, professional manner suitable for HR specialists.
    """

            synthesis_user_prompt = f"""
    Main query: {query}

    {synthesis_content}

    Please synthesize these answers into a cohesive response to the main query.
    """

            synthesis_prompt = AiServicePrompt(
                system_prompt=synthesis_system_prompt,
                user_prompt=synthesis_user_prompt
            )

            synthesis_result = self.ai_service.generate_response(
                response_model=ResultResponse,
                prompt=synthesis_prompt,
            )

            return synthesis_result.response
        except Exception as e:
            logger.error(f"Error generating multi-step response: {e}")
