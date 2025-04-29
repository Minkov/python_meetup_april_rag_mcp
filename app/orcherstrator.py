from typing import List, Dict, Any, Optional
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
    
    def __init__(
        self,
        retriever: Retriever,
        mcp: ModelContextProtocol,
        max_retrieval_rounds: int = 3,
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
        self.enable_logging = enable_logging
        self.query_analysis_service = QueryAnalysisService()
        
        # Keep track of already retrieved documents to avoid duplication
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
        # Track metrics for debugging and optimization
        total_retrievals = 0
        follow_up_queries = []
        
        # Phase 1: Initial Query Analysis and Retrieval
        logger.info(f"Initial query: {query}")
        
        # Analyze the query to understand information needs
        improved_query = self.query_analysis_service.improve_query(query)
        query_analysis = self.query_analysis_service.analyze_query(improved_query)
        
        # Initial retrieval based on query analysis
        retrieval_result = self.retriever.retrieve(
            query, 
            top_k=5,
            max_iterations=1,
        )
        
        retrieved_docs = retrieval_result["retrieved_docs"]
        total_retrievals += 1
        
        # Track retrieved document IDs
        self.retrieved_doc_ids.update([doc.doc_id for doc in retrieved_docs])
        
        logger.info(f"Initial retrieval: {len(retrieved_docs)} documents")
        
        # Phase 2: Reasoning and Follow-up Retrieval Loop
        for _ in range(self.max_retrieval_rounds):
            # Generate initial response based on current documents
            current_response = self.mcp.generate_response(
                query=query,
                context=retrieved_docs,
                previous_context=conversation_context
            )

            if current_response.is_complete:
                logger.info(f"MCP has provided a complete response")
                break
                
            # Log follow-up request
            logger.info(f"Follow-up request: {current_response.information_needs}")
            
            for need in current_response.information_needs:
                additional_docs = self._retrieve_additional_documents(
                    RetrievalRequest(
                        query=need.question,
                        context=need.context,
                        max_results=5,
                    )
                )
                
                if additional_docs:
                    logger.info(f"Retrieved {len(additional_docs)} additional documents")
                    retrieved_docs.extend(additional_docs)
                    total_retrievals += 1
                    
                    # Update tracking of retrieved documents
                    self.retrieved_doc_ids.update([doc.doc_id for doc in additional_docs])
                else:
                    logger.info("No additional relevant documents found")
                    break
        
        # Phase 3: Final Response Generation with all gathered information
        final_response = self.mcp.generate_response(
            query=query,
            context=retrieved_docs,
            previous_context=conversation_context,
            identify_information_needs=False
        )
        
        # Create and return the final result
        result = OrchestratorResult(
            response=final_response,
            retrieved_docs=retrieved_docs,
            follow_up_queries=follow_up_queries,
            total_retrievals=total_retrievals,
            query_analysis=query_analysis.model_dump()
        )
        
        return result
    
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