from typing import List, Dict, Any, Optional, Union, Callable, Type
import logging
import json
from enum import Enum

from openai import OpenAI
from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ai_service import AiService, AiServicePrompt
from app.schemas.base import AiResponseBaseModel
from app.schemas.document_results import DocumentResults

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


class ModelContextProtocol:
    """Enhanced ModelContextProtocol with advanced context handling and reasoning strategies."""

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 1000,
        temperature: float = 0.3,
        relevance_threshold: float = 4.0,
        enable_logging: bool = True
    ):
        """Initialize the ModelContextProtocol.

        Args:
            openai_api_key: API key for OpenAI
            model: Model to use for generation
            max_tokens: Maximum tokens in the response
            temperature: Temperature for generation
            relevance_threshold: Threshold for considering context relevant
            enable_logging: Whether to enable logging
        """
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        self.ai_service = AiService(openai_api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.relevance_threshold = relevance_threshold
        self.enable_logging = enable_logging

        # Token estimator function - can be replaced with more accurate one if needed
        self._token_estimator = lambda text: len(text) // 4

        # Base system prompts for different reasoning strategies
        self.system_prompts = {
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

    def _determine_reasoning_strategy(self, query: str) -> ReasoningStrategy:
        """Dynamically determine the best reasoning strategy for a query.

        Args:
            query: User query

        Returns:
            Recommended reasoning strategy
        """
        # Use AI to determine the most appropriate reasoning strategy
        prompt = f"""
You are an AI assistant that analyzes queries to determine the best reasoning strategy.

For a given query, determine the most appropriate reasoning strategy from the following options:
1. DIRECT - Simple, straightforward questions that can be answered directly from the context
2. COT (Chain-of-Thought) - Questions requiring step-by-step reasoning
3. MULTI_STEP - Complex questions that should be broken down into sub-questions

Return ONLY one of these three values: {", ".join([x.value for x in ReasoningStrategy])}
"""

        user_prompt = f"Analyze this query and determine the best reasoning strategy: {query}"

        prompt_obj = AiServicePrompt(
            system_prompt=prompt,
            user_prompt=user_prompt
        )

        class ReasoningStrategyResponse(AiResponseBaseModel):
            strategy: ReasoningStrategy = Field(
                description="The best reasoning strategy for the query",
                values=[x.value for x in ReasoningStrategy],
            )

        try:
            response = self.ai_service.generate_response(
                response_model=ReasoningStrategyResponse,
                prompt=prompt_obj,
                model=self.model,
                temperature=0.1
            )

            strategy_str = response.strategy.upper()

            if strategy_str == "DIRECT":
                return ReasoningStrategy.DIRECT
            elif strategy_str in ["COT", "CHAIN_OF_THOUGHT"]:
                return ReasoningStrategy.COT
            elif strategy_str in ["MULTI_STEP", "MULTISTEP"]:
                return ReasoningStrategy.MULTI_STEP
            else:
                # Default to direct if response is unexpected
                logger.warning(
                    f"Unexpected reasoning strategy: {strategy_str}. Using DIRECT.")
                return ReasoningStrategy.DIRECT

        except Exception as e:
            logger.error(f"Error determining reasoning strategy: {e}")
            return ReasoningStrategy.DIRECT

    def _determine_context_strategy(self, query: str, context: str) -> ContextStrategy:
        """Dynamically determine the best context strategy based on query and context.

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

    def generate_response(
        self,
        query: str,
        context: Union[str, List[DocumentResults]],
        reasoning_strategy: Optional[ReasoningStrategy] = None,
        context_strategy: Optional[ContextStrategy] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        structured_output_format: Optional[Type[AiResponseBaseModel]] = None
    ) -> Union[str, Dict[str, Any], AiResponseBaseModel]:
        """Generate a response to a query using the provided context.

        Args:
            query: User query
            context: Context information (string or DocumentResults)
            reasoning_strategy: Reasoning strategy to use (auto-determined if None)
            context_strategy: Context handling strategy to use (auto-determined if None)
            model: Model to use (overrides instance default)
            max_tokens: Max tokens (overrides instance default)
            temperature: Temperature (overrides instance default)
            structured_output_format: Optional pydantic model class for structured output

        Returns:
            Generated response as string, dict, or structured object
        """
        # Format context if it's a list of DocumentResults
        if isinstance(context, list):
            formatted_context = self._format_context(context)
        else:
            formatted_context = context

        # Dynamically determine strategies if not provided
        if reasoning_strategy is None:
            reasoning_strategy = self._determine_reasoning_strategy(query)
            logger.info(
                f"Dynamically selected reasoning strategy: {reasoning_strategy}")

        if context_strategy is None:
            context_strategy = self._determine_context_strategy(
                query, formatted_context)
            logger.info(
                f"Dynamically selected context strategy: {context_strategy}")

        # Set other defaults
        model = model or self.model
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Format context if it's a list of DocumentResults
        if isinstance(context, list):
            context = self._format_context(context)

        # Apply context strategy
        processed_context = self._apply_context_strategy(
            context, context_strategy, query)

        # Log context size estimate for debugging
        if self.enable_logging:
            context_tokens = self._token_estimator(processed_context)
            logger.info(f"Context size (estimated): {context_tokens} tokens")

        # Select appropriate reasoning handler based on strategy
        if reasoning_strategy == ReasoningStrategy.DIRECT:
            return self._direct_generation(query, processed_context, model, max_tokens, temperature, structured_output_format)
        elif reasoning_strategy == ReasoningStrategy.COT:
            return self._chain_of_thought_generation(query, processed_context, model, max_tokens, temperature, structured_output_format)
        elif reasoning_strategy == ReasoningStrategy.MULTI_STEP:
            return self._multi_step_generation(query, processed_context, model, max_tokens, temperature, structured_output_format)
        else:
            # Default to direct if unknown strategy
            logger.warning(
                f"Unknown reasoning strategy: {reasoning_strategy}. Using direct generation instead.")
            return self._direct_generation(query, processed_context, model, max_tokens, temperature, structured_output_format)

    def _format_context(self, documents: List[DocumentResults]) -> str:
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

    def _apply_context_strategy(
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
            # For GPT-4, roughly ~8k tokens for context is safe (model dependent)
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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

            compressed_context = response.choices[0].message.content

            # Add note about compression
            return compressed_context + "\n\n[Note: Context was compressed to preserve relevant information while reducing length.]"

        except Exception as e:
            logger.error(f"Error compressing context: {e}")
            # Fall back to truncation
            return self._apply_context_strategy(context, ContextStrategy.TRUNCATED, query)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _direct_generation(
        self,
        query: str,
        context: str,
        model: str,
        max_tokens: int,
        temperature: float,
        structured_output_format: Optional[Type[AiResponseBaseModel]] = None
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

        # If using structured output, use the AiService with the provided schema
        if structured_output_format:
            try:
                result = self.ai_service.generate_response(
                    response_model=structured_output_format,
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return result
            except Exception as e:
                logger.error(f"Error generating structured response: {e}")
                # Fall back to unstructured response
                result = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return {"error": "Failed to parse structured output",
                        "raw_response": result.choices[0].message.content}
        else:
            # Use direct OpenAI client for unstructured responses
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _chain_of_thought_generation(
        self,
        query: str,
        context: str,
        model: str,
        max_tokens: int,
        temperature: float,
        structured_output_format: Optional[Type[AiResponseBaseModel]] = None
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

        # Handle structured output if specified
        if structured_output_format:
            # For structured output with CoT, we'll do a two-step process
            # First, generate the reasoning
            reasoning_prompt = AiServicePrompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # Use direct OpenAI client for the reasoning step
            reasoning_response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            reasoning = reasoning_response.choices[0].message.content

            # Then, create structured output using AiService
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
                    response_model=structured_output_format,
                    prompt=final_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens // 2
                )
                return result
            except Exception as e:
                logger.error(f"Error generating structured response: {e}")
                return {"error": "Failed to parse structured output",
                        "reasoning": reasoning}
        else:
            # For unstructured output, just return the reasoning response
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _multi_step_generation(
        self,
        query: str,
        context: str,
        model: str,
        max_tokens: int,
        temperature: float,
        structured_output_format: Optional[Type[AiResponseBaseModel]] = None
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

        class SubQuestionsResponse(AiResponseBaseModel):
            sub_questions: List[str]

        try:
            subq_result = self.ai_service.generate_response(
                response_model=SubQuestionsResponse,
                prompt=subq_prompt,
                model=self.model,
                temperature=0.3,
                max_tokens=500
            )
            sub_questions = subq_result.sub_questions
        except Exception as e:
            logger.warning(f"Error parsing sub-questions with AiService: {e}")
            # Use direct OpenAI client as fallback
            subq_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": subq_system_prompt},
                    {"role": "user", "content": subq_user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            sub_questions_text = subq_response.choices[0].message.content

            # Extract sub-questions using simple line parsing
            sub_questions = []
            for line in sub_questions_text.split('\n'):
                line = line.strip()
                # Check if line starts with a number followed by period/bracket/etc.
                if line and (line[0].isdigit() or (line[0] == '-')):
                    # Remove the number prefix
                    question = line[1:].strip()
                    if line[1:3] in ['. ', ') ', '- ', ': ']:
                        question = line[3:].strip()
                    if question:
                        sub_questions.append(question)

        if not sub_questions:
            # Fall back to direct generation if parsing failed
            logger.warning(
                "Failed to parse sub-questions, falling back to direct generation")
            return self._direct_generation(query, context, model, max_tokens, temperature, structured_output_format)

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

            # Use direct OpenAI client for sub-answers (faster)
            sub_response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sub_system_prompt},
                    {"role": "user", "content": sub_user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens // 2  # Divide tokens to allow for final synthesis
            )

            sub_answers.append({
                "question": sub_q,
                "answer": sub_response.choices[0].message.content
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

        # Handle structured output if specified
        if structured_output_format:
            # First generate synthesis with direct OpenAI call
            synthesis_response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": synthesis_system_prompt},
                    {"role": "user", "content": synthesis_user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            synthesis = synthesis_response.choices[0].message.content

            # Then use AiService to create structured output
            final_system_prompt = f"""
{synthesis_system_prompt}

You previously synthesized the following response:
{synthesis}

Now, provide your final answer as a structured output in the format requested.
"""

            final_user_prompt = "Based on your synthesis, provide the final structured answer."

            final_prompt = AiServicePrompt(
                system_prompt=final_system_prompt,
                user_prompt=final_user_prompt
            )

            try:
                result = self.ai_service.generate_response(
                    response_model=structured_output_format,
                    prompt=final_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens // 2
                )
                return result
            except Exception as e:
                logger.error(f"Error generating structured response: {e}")
                return {"error": "Failed to parse structured output",
                        "synthesis": synthesis}
        else:
            # For unstructured output, just return the synthesis
            synthesis_response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": synthesis_system_prompt},
                    {"role": "user", "content": synthesis_user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return synthesis_response.choices[0].message.content

    def assess_context_relevance(self, query: str, context: Union[str, List[DocumentResults]]) -> Dict[str, Any]:
        """Assess if the context is relevant and sufficient for answering the query.

        Args:
            query: User query
            context: Context information

        Returns:
            Dictionary with relevance assessment
        """
        # Format context if it's a list of DocumentResults
        if isinstance(context, list):
            context = self._format_context(context)

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
                model=self.model,
                temperature=0.3
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

    def set_token_estimator(self, estimator_func: Callable[[str], int]):
        """Set a custom token estimator function.

        Args:
            estimator_func: Function that takes a string and returns estimated token count
        """
        self._token_estimator = estimator_func
