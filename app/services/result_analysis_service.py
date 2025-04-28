from typing import List, Dict, Any
from openai import OpenAI

from app.ai_service import AiServiceCapability, AiServicePrompt, get_ai_service
from app.schemas.retriever import InformationGapType, RetrieverQueryAnalysis, RetrieverResultsAnalysis, RetrieverOptimizedQueries, RetrieverDocScoresResult, ResultAnswer
from app.schemas.document_results import DocumentResults

class ResultAnalysisService:
    def __init__(self):
        """Initialize the ResultAnalysisService

        Args:
            model: LLM model to use
        """
        self.ai_service = get_ai_service(AiServiceCapability.FAST)

    def analyze_results(self, query: str, query_analysis: RetrieverQueryAnalysis, formatted_results: str) -> RetrieverResultsAnalysis:
        """Analyze retrieved results to determine if more information is needed

        Args:
            query: Original user query
            query_analysis: RetrieverQueryAnalysis object
            formatted_results: Formatted results string

        Returns:
            RetrieverResultsAnalysis object
        """
        prompt = """
You are an HR Intelligence Assistant analyzing documents retrieved from ChromaDB, a vector database using semantic similarity search.

You are analyzing results from a hierarchical document system:
- Base level: Specific details and facts (300-500 tokens)
- Section level: Topical sections with more context (800-1500 tokens)
- Document level: Complete documents with full context

When analyzing the results, consider:
1. How well the retrieved documents match the semantic intent of the query
2. Whether the results contain the specific information needed
3. The relevance of the hierarchy level to the query type
4. The coverage of key entities and terms identified in the query analysis

Respond *only* with a valid JSON object matching the provided schema. Do not add any introductory text, explanations outside the JSON structure, or markdown formatting.
"""
        user_prompt = f'''
# Input
User query: {query}

Query analysis: {query_analysis.model_dump_json()}

Retrieved documents from ChromaDB:
{formatted_results}

Your task is to:
1. Analyze if these documents fully answer the user's query
2. Consider that information might be spread across documents with varying similarity scores
3. Identify specific information gaps (if any)
4. Focus on the most relevant information needs from the query analysis

If the combined documents contain ALL the information needed to fully answer the query, 
respond with "COMPLETE: [brief explanation why the information is complete]"

If more information is needed, respond with:
- "INCOMPLETE: [explanation of what's missing]"
- Then list specific information gaps. Possible values are: {[x.value for x in InformationGapType]}

Be precise in identifying missing information that would be critical for answering the original query.
'''
        prompt = AiServicePrompt(
            system_prompt=prompt,
            user_prompt=user_prompt
        )

        result = self.ai_service.generate_response(
            prompt=prompt, response_model=RetrieverResultsAnalysis)
        return result

    def generate_follow_up_queries(
        self,
        query: str,
        query_analysis: RetrieverQueryAnalysis,
        analysis_result: RetrieverResultsAnalysis,
    ) -> RetrieverOptimizedQueries:
        """Generate follow-up queries for additional information

        Args:
            query: Original user query
            query_analysis: RetrieverQueryAnalysis object
            analysis_result: RetrieverResultsAnalysis object

        Returns:
            RetrieverOptimizedQueries object
        """
        # If analysis indicates completeness, no follow-up queries needed
        if analysis_result.status == ResultAnswer.COMPLETE:
            return RetrieverOptimizedQueries(optimized_queries=[])

        prompt = """
You are an HR Intelligence Assistant generating follow-up queries for ChromaDB, a vector database that uses semantic similarity for retrieval.

CHROMADB OPTIMIZATION GUIDELINES:
1. Create concise queries (4-8 words is ideal)
2. Focus on specific missing information identified in the analysis
3. Use high-signal HR terminology likely to appear in relevant documents
4. Structure queries from most to least important terms
5. Remove low-value words (articles, common verbs) that dilute the embedding

Respond *only* with a valid JSON object matching the provided schema. Do not add any introductory text, explanations outside the JSON structure, or markdown formatting.
"""
        user_prompt = f'''
# Input
Original user query: {query}

Query analysis: {query_analysis.model_dump_json()}

Analysis of current results: {analysis_result.model_dump_json()}

Based on the identified information gaps, generate 1-3 follow-up queries optimized for ChromaDB.
Each query should target one specific missing information need with precision.

For each information gap:
1. Create a concise, keyword-rich query (4-8 words ideal)
2. Focus on domain-specific terminology that would appear in target documents
3. Structure from most to least important concepts
4. Exclude filler words that add little semantic value

Provide ONLY the optimized queries, with no explanations or additional text.
'''

        prompt = AiServicePrompt(
            system_prompt=prompt,
            user_prompt=user_prompt
        )

        result = self.ai_service.generate_response(
            prompt=prompt, response_model=RetrieverOptimizedQueries)
        return result

    def score_results(self, query: str, query_analysis: RetrieverQueryAnalysis, results: List[DocumentResults]) -> RetrieverDocScoresResult:
        """Rank results based on query analysis to get the most relevant documents,
        with awareness of hierarchy levels

        Args:
            query: Original user query
            query_analysis: RetrieverQueryAnalysis object
            results: List of DocumentResults

        Returns:
            RetrieverDocScoresResult object with ranked documents
        """
        prompt = """
You are an HR Intelligence Assistant ranking document chunks retrieved from ChromaDB, a vector database using semantic similarity search.

Effective ChromaDB result scoring considers:
1. Semantic similarity to the query intent (not just keyword matching)
2. Presence of relevant entities and domain-specific terminology
3. Information density relevant to the query topic
4. Document hierarchy level appropriateness for the query type

Score each result's relevance on a scale from 1-10:
- High scores (8-10): Directly answers the query with specific relevant information
- Medium scores (5-7): Contains partially relevant information or context
- Low scores (1-4): Tangentially relevant or contains minimal useful information

When scoring, consider:
- For specific fact queries: Base level chunks with direct answers score higher
- For context-heavy queries: Section or document level chunks may score higher
- Presence of key entities identified in query analysis significantly increases score

Respond *only* with a valid JSON object matching the provided schema.
"""

        formatted_results = self._format_results_for_analysis(results)

        user_prompt = f'''
# Input
Original query: {query}

Query analysis: {query_analysis.model_dump_json()}

Retrieved document chunks from ChromaDB:
{formatted_results}

# Instructions
Rank the document chunks based on:
1. Semantic relevance to the query intent
2. Presence of key entities and terminology identified in query analysis
3. Information density and specificity relevant to the query
4. Appropriate hierarchy level for the query type

Provide the document IDs, relevance scores (1-10), and explain your scoring criteria.
        '''

        prompt = AiServicePrompt(
            system_prompt=prompt,
            user_prompt=user_prompt
        )

        result = self.ai_service.generate_response(
            prompt=prompt, response_model=RetrieverDocScoresResult)
        return result

    def _format_results_for_analysis(self, results: List[DocumentResults]) -> str:
        """Format results for the analysis step, with hierarchy awareness

        Args:
            results: List of DocumentResults

        Returns:
            Formatted results string
        """
        formatted_results = []
        for i, result in enumerate(results, start=1):
            parent_id = self._get_parent_id(result)
            hierarchy_level = result.metadata.get("hierarchy_level", "unknown")

            formatted_results.append(f"Document {i}:")
            formatted_results.append(f"Document ID: {result.doc_id}")
            formatted_results.append(f"Parent ID: {parent_id}")
            formatted_results.append(f"Hierarchy Level: {hierarchy_level}")
            formatted_results.append(f"Similarity Score: {result.similarity:.4f}")

            # Add metadata if available
            if result.metadata:
                relevant_metadata = {k: v for k, v in result.metadata.items()
                                     if k not in ['hierarchy_level', 'parent_id', 'base_chunks',
                                                  'section_chunks', 'hierarchy_index', 'token_count']}
                if relevant_metadata:
                    formatted_results.append("Metadata:")
                    for key, value in relevant_metadata.items():
                        formatted_results.append(f"- {key}: {value}")

            # Add document content
            formatted_results.append("Content:")
            formatted_results.append(result.text)
            formatted_results.append("-" * 40)  # Separator

        return "\n".join(formatted_results)

    def _get_parent_id(self, result: DocumentResults) -> str:
        """Extract parent document ID from a result

        Args:
            result: DocumentResults object

        Returns:
            Parent ID string
        """
        # First try to get from metadata
        if result.metadata and 'parent_id' in result.metadata:
            return result.metadata['parent_id']

        # Fall back to parsing from doc_id if it follows our convention
        doc_id = result.doc_id

        # Handle various ID formats
        if '_chunk_' in doc_id:
            # Format: {parent_id}_chunk_{chunk_index}
            return doc_id.split('_chunk_')[0]
        elif '_base_' in doc_id:
            # Format: {parent_id}_base_{chunk_index}
            return doc_id.split('_base_')[0]
        elif '_section_' in doc_id:
            # Format: {parent_id}_section_{section_index}
            return doc_id.split('_section_')[0]

        # If we can't determine parent, use the doc_id itself
        return doc_id