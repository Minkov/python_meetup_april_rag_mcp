import logging

from app.ai_service import AiServiceCapability, get_ai_service
from app.schemas.document_results import DocumentResults
from app.schemas.retriever import DocumentType, InformationGapType, ResultAnswer, RetrieverQueryAnalysis
from app.vector_db import VectorDatabase
from app.services.query_analysis_service import QueryAnalysisService
from app.services.result_analysis_service import ResultAnalysisService
from app.services.query_complexity_service import QueryComplexityService

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultistageRetriever:
    def __init__(
        self,
        vector_db: VectorDatabase,
        relevance_threshold: float = 4.0,
        enable_caching: bool = True
    ):
        self.vector_db = vector_db
        self.fast_ai_service = get_ai_service(AiServiceCapability.FAST)
        self.thinking_ai_service = get_ai_service(AiServiceCapability.THINKING)
        self.query_analysis_service = QueryAnalysisService()
        self.result_analysis_service = ResultAnalysisService()
        self.query_complexity_service = QueryComplexityService()

        self.relevance_threshold = relevance_threshold
        self.enable_caching = enable_caching

    def retrieve(self, query, top_k=5, max_iterations=3):
        improved_query = self.query_analysis_service.improve_query(query)

        query_analysis = self.query_analysis_service.analyze_query(
            improved_query)
        
        optimized_queries = [
            *self._people_queries(improved_query, query_analysis),
            *self._team_queries(improved_query, query_analysis),
            *self._job_queries(improved_query, query_analysis),
            *self._interview_queries(improved_query, query_analysis),
        ]

        results = []
        for optimized_query in optimized_queries:
            query_results = self._get_query_results(
                optimized_query.optimized_query,
                query_analysis,
                top_k
            )
            results.extend(query_results)

        selected_results = self._select_best_results(results, improved_query, query_analysis)

        iteration_count = 0
        while iteration_count < max_iterations:
            logger.info(
                f"Starting iteration {iteration_count + 1}/{max_iterations}")

            additional_results = self._make_iteration(
                improved_query, query_analysis, selected_results)

            if not additional_results:
                logger.info(
                    f"No additional results found in iteration {iteration_count + 1}. Stopping.")
                break

            selected_results.extend(additional_results)
            selected_results.sort(key=lambda x: x.similarity, reverse=True)

            if len(selected_results) > top_k * 2:
                selected_results = selected_results

            logger.info(
                f"After iteration {iteration_count + 1}: {len(selected_results)} documents")
            iteration_count += 1

        enhanced_results = self._enhance_with_hierarchy(selected_results)

        return {
            "retrieved_docs": enhanced_results,
            "query_analysis": query_analysis,
            "optimized_queries": optimized_queries,
        }

    def _people_queries(self, improved_query: str, query_analysis: RetrieverQueryAnalysis):
        if not query_analysis.people_identified:
            return []

        result = self.query_analysis_service.generate_person_queries(
            improved_query, query_analysis)

        return result.optimized_queries

    def _team_queries(self, improved_query: str, query_analysis: RetrieverQueryAnalysis):
        if not query_analysis.teams_identified:
            return []

        result = self.query_analysis_service.generate_team_queries(
            improved_query, query_analysis)

        return result.optimized_queries

    def _job_queries(self, improved_query: str, query_analysis: RetrieverQueryAnalysis):
        if not query_analysis.jobs_identified:
            return []

        result = self.query_analysis_service.generate_job_queries(
            improved_query, query_analysis)

        return result.optimized_queries

    def _interview_queries(self, improved_query: str, query_analysis: RetrieverQueryAnalysis):
        if not query_analysis.people_identified:
            return []

        result = self.query_analysis_service.generate_interview_queries(
            improved_query, query_analysis)

        return result.optimized_queries

    def _select_best_results(self, results, improved_query, query_analysis):
        """
        Select the best results by deduplicating and scoring them.
        
        Args:
            results: List of search results
            improved_query: The improved query text
            query_analysis: The query analysis object
            
        Returns:
            List of selected results that meet the relevance threshold
        """
        unique_results = {}
        for result in results:
            dedup_id = self._get_base_id(result)
            if dedup_id not in unique_results or result.similarity > unique_results[dedup_id].similarity:
                unique_results[dedup_id] = result

        scored_results = self.result_analysis_service.score_results(
            improved_query, query_analysis, list(unique_results.values()))

        selected_results = []
        for scored_doc in scored_results.documents:
            if scored_doc.relevance_score >= self.relevance_threshold:
                original_doc = next(
                    doc for doc in results
                    if scored_doc.doc_id == doc.doc_id
                )
                selected_results.append(original_doc)

        return selected_results

    def _get_query_results(self, query, query_analysis, top_k, filter_criteria=None):
        """
        Process a single optimized query and return search results.
        
        Args:
            query: The optimized query to process
            query_analysis: The query analysis object
            top_k: Number of top results to retrieve
            
        Returns:
            List of search results
        """
        query_complexity = self.query_complexity_service.estimate_query_complexity(query, query_analysis)
        query_hierarchy_level = self.query_complexity_service.select_hierarchy_level(query_complexity)

        logger.info(
            f"Using hierarchy level '{query_hierarchy_level}' and filter criteria '{filter_criteria}' for query: {query}")

        return self.vector_db.search(
            query,
            top_k=top_k * 2,
            hierarchy_preference=query_hierarchy_level,
            filter_criteria=filter_criteria,
        )

    def _make_iteration(self, improved_query, query_analysis, selected_results):
        """
        Performs an iteration of retrieval to find additional relevant information.

        Args:
            improved_query: The improved query text
            query_analysis: The query analysis object
            selected_results: Currently selected results

        Returns:
            List of additional relevant documents to add to results
        """
        if not selected_results:
            return []

        formatted_results = self.result_analysis_service._format_results_for_analysis(
            selected_results)

        analysis_result = self.result_analysis_service.analyze_results(
            improved_query, query_analysis, formatted_results)

        if analysis_result.status == ResultAnswer.COMPLETE:
            logger.info(
                f"Results are complete. Analysis: {analysis_result.explanation}")
            return []

        logger.info(
            f"Results are incomplete. Generating follow-up queries. Gaps: {analysis_result.information_gaps}")
        
        information_gaps = {x.information_type for x in analysis_result.information_gaps}
        follow_up_queries = []
        for information_gap in information_gaps:
            if information_gap == InformationGapType.JOB_OPENING:
                query_result = self.query_analysis_service.generate_job_queries(
                    improved_query, query_analysis)
                if query_result.optimized_queries:
                    follow_up_queries.append({
                        "queries": query_result.optimized_queries,
                        "filter_criteria": {
                            "document_type": DocumentType.JOB_OPENING.value
                        }
                    })
            elif information_gap == InformationGapType.TEAM_STRUCTURE:
                query_result = self.query_analysis_service.generate_team_queries(
                    improved_query, query_analysis)
                if query_result.optimized_queries:
                    follow_up_queries.append({
                        "queries": query_result.optimized_queries,
                        "filter_criteria": {
                            "document_type": DocumentType.TEAM_STRUCTURE.value
                        }
                    })
            elif information_gap == InformationGapType.RESUME:
                query_result = self.query_analysis_service.generate_person_queries(
                    improved_query, query_analysis)
                if query_result.optimized_queries:
                    follow_up_queries.append({
                        "queries": query_result.optimized_queries,
                        "filter_criteria": {
                            "document_type": DocumentType.RESUME.value
                        }
                    })
            elif information_gap == InformationGapType.MEETING_NOTES:
                query_result = self.query_analysis_service.generate_interview_queries(
                    improved_query, query_analysis)
                if query_result.optimized_queries:
                    follow_up_queries.append({
                        "queries": query_result.optimized_queries,
                        "filter_criteria": {
                            "document_type": DocumentType.MEETING_NOTES.value
                        }
                    })

        if not follow_up_queries:
            logger.info("No follow-up queries generated.")
            return []

        seen_doc_ids = {self._get_base_id(result)
                        for result in selected_results}

        additional_results = []

        for follow_up_queries_by_type in follow_up_queries:
            for follow_up_query in follow_up_queries_by_type["queries"]:
                query_results = self._get_query_results(
                    follow_up_query.optimized_query,
                    query_analysis,
                    top_k=10,
                    filter_criteria=follow_up_queries_by_type["filter_criteria"]
                )

                new_results = [
                    result for result in query_results
                    if self._get_base_id(result) not in seen_doc_ids
                ]

                additional_results.extend(new_results)
                seen_doc_ids.update(self._get_base_id(result) for result in new_results)

        if additional_results:
            # Use _select_best_results to score and filter the additional results
            relevant_additional = self._select_best_results(
                additional_results,
                improved_query,
                query_analysis
            )
            logger.info(
                f"Found {len(relevant_additional)} new relevant documents in iteration")
            return relevant_additional

        return []

    def _enhance_with_hierarchy(self, results):
        """
        Simplified method that enhances results by:
        1. Reconstructing document content when needed
        2. Adding related context from different hierarchy levels
        """
        enhanced_results = []
        added_ids = set()  # Track what we've already added to avoid duplicates

        # Process each result in the original order
        for result in results:
            # Skip if we've already added this result
            if result.doc_id in added_ids:
                continue

            # STEP 1: Handle document-level results - reconstruct content if needed
            if result.metadata.get("hierarchy_level") == "document":
                # Check if this is just a summary without full content
                if "Document summary for" in result.text:
                    # Try to reconstruct from sections
                    reconstructed_text = self._reconstruct_document_from_sections(
                        result)
                    if reconstructed_text:
                        # Create a new result with the reconstructed text
                        result = DocumentResults(
                            doc_id=result.doc_id,
                            similarity=result.similarity,
                            text=reconstructed_text,
                            metadata=result.metadata
                        )

                # Add the document result (original or reconstructed)
                enhanced_results.append(result)
                added_ids.add(result.doc_id)

                # Track any sections that are part of this document
                if result.metadata.get("section_chunks"):
                    for section_id in result.metadata.get("section_chunks").split(","):
                        added_ids.add(section_id)

                # Track any base chunks that are part of this document
                if result.metadata.get("base_chunks"):
                    for base_id in result.metadata.get("base_chunks").split(","):
                        added_ids.add(base_id)

            # STEP 2: Handle section-level results
            elif result.metadata.get("hierarchy_level") == "section":
                enhanced_results.append(result)
                added_ids.add(result.doc_id)

                # Track any base chunks that are part of this section
                if result.metadata.get("base_chunks"):
                    for base_id in result.metadata.get("base_chunks").split(","):
                        added_ids.add(base_id)

            # STEP 3: Handle base-level results
            elif result.metadata.get("hierarchy_level") == "base":
                enhanced_results.append(result)
                added_ids.add(result.doc_id)

                # Add adjacent chunks for context
                self._add_adjacent_chunks(result, enhanced_results, added_ids)

                # Try to find and add parent section if not already included
                self._add_parent_section(result, enhanced_results, added_ids)

        # Sort by similarity
        enhanced_results.sort(key=lambda x: x.similarity, reverse=True)

        return enhanced_results

    def _reconstruct_document_from_sections(self, doc_result):
        """Helper method to reconstruct document content from its sections"""
        if not doc_result.metadata.get("section_chunks"):
            return None

        # Get section IDs
        section_ids = doc_result.metadata.get("section_chunks").split(",")

        # Retrieve sections
        section_chunks = self.vector_db.collection.get(ids=section_ids)

        if section_chunks and section_chunks["ids"]:
            # Combine section texts
            return "\n\n".join(section_chunks["documents"])

        # If sections not found, try base chunks
        if doc_result.metadata.get("base_chunks"):
            base_ids = doc_result.metadata.get("base_chunks").split(",")
            base_chunks = self.vector_db.collection.get(ids=base_ids)

            if base_chunks and base_chunks["ids"]:
                return "\n\n".join(base_chunks["documents"])

        return None

    def _add_adjacent_chunks(self, result, enhanced_results, added_ids):
        """Helper method to add adjacent chunks for context"""
        adj_ids = []

        # Check for next and previous chunks
        if result.metadata.get("next_chunk_id") and result.metadata.get("next_chunk_id") not in added_ids:
            adj_ids.append(result.metadata["next_chunk_id"])

        if result.metadata.get("prev_chunk_id") and result.metadata.get("prev_chunk_id") not in added_ids:
            adj_ids.append(result.metadata["prev_chunk_id"])

        if not adj_ids:
            return

        # Retrieve adjacent chunks
        adj_results = self.vector_db.collection.get(ids=adj_ids)

        if adj_results and adj_results["ids"]:
            for i, adj_id in enumerate(adj_results["ids"]):
                if adj_id not in added_ids:
                    # Add with slightly lower similarity to indicate contextual relationship
                    enhanced_results.append(DocumentResults(
                        doc_id=adj_id,
                        similarity=result.similarity * 0.85,
                        text=adj_results["documents"][i],
                        metadata=adj_results["metadatas"][i]
                    ))
                    added_ids.add(adj_id)

    def _add_parent_section(self, result, enhanced_results, added_ids):
        """Helper method to add parent section for context"""
        doc_id = result.metadata.get("parent_id", "")
        if not doc_id:
            return

        # Find section containing this chunk
        section_query = self.vector_db.collection.get(
            where={
                "$and": [
                    {"hierarchy_level": "section"},
                    {"parent_id": doc_id},
                    {"base_chunks": {"$in": [result.doc_id]}}
                ]
            }
        )

        if section_query and section_query["ids"]:
            for i, section_id in enumerate(section_query["ids"]):
                if section_id not in added_ids:
                    enhanced_results.append(DocumentResults(
                        doc_id=section_id,
                        similarity=result.similarity * 0.9,
                        text=section_query["documents"][i],
                        metadata=section_query["metadatas"][i]
                    ))
                    added_ids.add(section_id)

    def _get_base_id(self, result):
        if result.metadata and "parent_id" in result.metadata:
            return result.metadata["parent_id"]

        doc_id = result.doc_id

        for prefix in ["_base_", "_section_", "_document"]:
            if prefix in doc_id:
                return doc_id.split(prefix)[0]

        return doc_id

    def format_hierarchical_context(self, results):
        if not results:
            return "No relevant information found."

        context_parts = []
        parent_ids_included = set()

        for i, result in enumerate(results, start=1):
            parent_id = result.metadata.get("parent_id", "unknown")
            hierarchy_level = result.metadata.get("hierarchy_level", "unknown")

            # Add separator between different documents
            is_first = (i == 1)
            new_parent = parent_id not in parent_ids_included

            if not is_first and new_parent:
                context_parts.append("\n" + "-" * 40 + "\n")  # Separator

            parent_ids_included.add(parent_id)

            # Format header based on hierarchy level
            if hierarchy_level == "section":
                title = result.metadata.get(
                    "title", f"Section from document {parent_id}")
                context_parts.append(f"Document {i} (Section: {title}):")
            elif hierarchy_level == "base":
                context_parts.append(
                    f"Document {i} (Detail chunk from {parent_id}):")
            elif hierarchy_level == "document":
                context_parts.append(f"Document {i} (Complete document):")
            else:
                context_parts.append(f"Document {i}:")

            context_parts.append(f"Document ID: {result.doc_id}")
            context_parts.append(f"Relevance: {result.similarity:.4f}")

            # Add relevant metadata
            relevant_metadata = {k: v for k, v in result.metadata.items()
                                 if k not in ['hierarchy_level', 'parent_id', 'base_chunks',
                                              'section_chunks', 'hierarchy_index', 'token_count',
                                              'next_chunk_id', 'prev_chunk_id']}

            if relevant_metadata:
                context_parts.append("Metadata:")
                for key, value in relevant_metadata.items():
                    context_parts.append(f"- {key}: {value}")

            # Add document content
            context_parts.append("Content:")
            context_parts.append(result.text)
            context_parts.append("-" * 40)  # Separator

        return "\n".join(context_parts)
