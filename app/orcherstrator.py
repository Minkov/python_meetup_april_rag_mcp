from typing import List, Dict, Any, Optional, Tuple
import logging

from app.mcp import ModelContextProtocol
from app.retriever import Retriever
from app.schemas.document_results import DocumentResults
from app.schemas.orchestrator_schemas import RetrievalRequest, OrchestratorResult
from app.services.query_analysis_service import QueryAnalysisService

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Orchestrates the interaction between reasoning (MCP) and retrieval components,
    enabling bidirectional communication for optimal information gathering.
    """

    MAX_RETRIEVAL_ROUNDS = 3
    MAX_RETRIEVAL_DOCS = 5

    def __init__(
        self,
        retriever: Retriever,
        mcp: ModelContextProtocol,
        max_retrieval_rounds: int = MAX_RETRIEVAL_ROUNDS,
        max_retrieval_docs: int = MAX_RETRIEVAL_DOCS,
        enable_logging: bool = True
    ):
        """Initialize the RAG Orchestrator.

        Args:
            retriever: The retrieval component
            mcp: The Model-Context Protocol component
            max_retrieval_rounds: Maximum number of retrieval rounds
            enable_logging: Whether to enable detailed logging
        """
        self.retriever = retriever
        self.mcp = mcp
        self.max_retrieval_rounds = max_retrieval_rounds
        self.max_retrieval_docs = max_retrieval_docs
        self.enable_logging = enable_logging
        self.query_analysis_service = QueryAnalysisService()

        self.retrieved_doc_ids = set()

    def process_query(self, query: str, conversation_context: Optional[str] = None) -> OrchestratorResult:
        """
        Process a user query through the RAG pipeline with bidirectional 
        communication between reasoning and retrieval.

        Args:
            query: The user's query
            conversation_context: Optional previous conversation context

        Returns:
            OrchestratorResult with the final response and debug information
        """
        total_retrievals = 0

        logger.info(f"Initial query: {query}")

        retrieval_result = self.retriever.retrieve(
            query,
            top_k=self.max_retrieval_docs,
            max_iterations=self.max_retrieval_rounds,
        )

        retrieved_docs = retrieval_result["retrieved_docs"]

        self.retrieved_doc_ids.update([doc.doc_id for doc in retrieved_docs])

        logger.info(f"Initial retrieval: {len(retrieved_docs)} documents")

        retrieved_docs, total_retrievals = self._process_retrieval_iterations(
            query=query,
            retrieved_docs=retrieved_docs,
            conversation_context=conversation_context
        )

        logger.info(f"Final retrieval: {len(retrieved_docs)} documents")

        final_response = self.mcp.generate_response(
            query=query,
            context=retrieved_docs,
            previous_context=conversation_context,
            identify_information_needs=False
        )

        result = OrchestratorResult(
            response=final_response,
            retrieved_docs=retrieved_docs,
            total_retrievals=total_retrievals,
        )

        return result

    def _process_retrieval_iterations(
        self,
        query: str,
        retrieved_docs: List[DocumentResults],
        conversation_context: Optional[str] = None
    ) -> Tuple[List[DocumentResults], int]:
        """
        Process multiple rounds of retrieval based on information needs.

        Args:
            query: The original query
            retrieved_docs: Initially retrieved documents
            conversation_context: Optional previous conversation context

        Returns:
            Updated list of retrieved documents
        """
        new_retrieved_docs = [x for x in retrieved_docs]
        total_retrievals = 0

        for _ in range(self.max_retrieval_rounds):
            total_retrievals += 1
            result = self._process_retrieval_iteration(
                query=query,
                retrieved_docs=retrieved_docs,
                conversation_context=conversation_context
            )
            if result:
                new_retrieved_docs.extend(result)
            else:
                break

        return new_retrieved_docs, total_retrievals

    def _process_retrieval_iteration(
        self,
        query: str,
        retrieved_docs: List[DocumentResults],
        conversation_context: Optional[str] = None
    ) -> List[DocumentResults]:
        current_response = self.mcp.generate_response(
            query=query,
            context=retrieved_docs,
            previous_context=conversation_context
        )

        if current_response.is_complete:
            logger.info(f"No more information needs to be retrieved")
            return retrieved_docs
        new_retrieved_docs = []

        for need in current_response.information_needs:
            additional_docs = self._retrieve_additional_documents(
                RetrievalRequest(
                    query=need.question,
                    context=need.context,
                    max_results=5,
                )
            )

            if additional_docs:
                logger.info(
                    f"Retrieved {len(additional_docs)} additional documents")
                new_retrieved_docs.extend(additional_docs)

                self.retrieved_doc_ids.update(
                    [doc.doc_id for doc in additional_docs])
            else:
                logger.info("No additional relevant documents found")
                break
        return new_retrieved_docs

  
    def _retrieve_additional_documents(self, request: RetrievalRequest) -> List[DocumentResults]:
        """
        Retrieve additional documents based on a follow-up request.

        Args:
            request: The retrieval request

        Returns:
            List of additional DocumentResults
        """
        # Use the retriever to get documents
        retrieval_result = self.retriever.retrieve(
            request.query,
            top_k=request.max_results,
            max_iterations=1,
        )

        documents = retrieval_result["retrieved_docs"]

        # Filter out documents we've already retrieved
        new_documents = [
            doc for doc in documents
            if doc.doc_id not in self.retrieved_doc_ids
        ]

        return new_documents

    def get_conversation_context(self, query: str, retrieved_docs: List[DocumentResults]) -> str:
        """
        Create a formatted conversation context from retrieved documents.

        Args:
            query: The original query
            retrieved_docs: Retrieved documents

        Returns:
            Formatted context string
        """
        return self.retriever.format_hierarchical_context(retrieved_docs)
