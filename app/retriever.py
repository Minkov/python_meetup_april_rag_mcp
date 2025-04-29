import logging
from typing import List

from app.ai_service import AiServiceCapability, get_ai_service
from app.schemas.document_results import DocumentResults
from app.schemas.retriever import DocumentType, InformationGapType, ResultAnswer, RetrieverOptimizedQuery, RetrieverQueryAnalysis
from app.vector_db import VectorDatabase
from app.services.query_analysis_service import QueryAnalysisService
from app.services.result_analysis_service import ResultAnalysisService
from app.services.query_complexity_service import QueryComplexityService

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Retriever:
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
        # Step 1: Improve the query
        improved_query = self.query_analysis_service.improve_query(query)

        # Step 2: Analyze the query
        query_analysis = self.query_analysis_service.analyze_query(
            improved_query)

        # Step 3: Split the query into multiple optimized queries
        optimized_queries = self._get_optimized_queries(
            improved_query, query_analysis)

        # Step 4: Get initial results
        results = self._get_initial_results(
            optimized_queries, query_analysis, top_k)

        # Step 5: Select the best results
        selected_results = self._select_best_results(
            results, improved_query, query_analysis)

        # Step 6: Iterate to find additional relevant information
        extended_results = self._extend_results(
            selected_results, improved_query, query_analysis, max_iterations)

        # Step 7: Enhance the results with hierarchy
        enhanced_results = self._enhance_with_hierarchy(extended_results)

        # Step 8: Return the results
        return {
            "retrieved_docs": enhanced_results,
            "query_analysis": query_analysis,
            "optimized_queries": optimized_queries,
        }

    def _extend_results(self, selected_results, improved_query, query_analysis, max_iterations):
        current_results = [x for x in selected_results]
        for iteration_count in range(max_iterations):
            logger.info(
                f"Starting iteration {iteration_count + 1}/{max_iterations}")

            additional_results = self._perform_iteration(
                improved_query, query_analysis, current_results)

            if not additional_results:
                logger.info(
                    f"No additional results found in iteration {iteration_count + 1}. Stopping.")
                break

            current_results.extend(additional_results)
            current_results.sort(key=lambda x: x.similarity, reverse=True)

            logger.info(
                f"After iteration {iteration_count + 1}: {len(current_results)} documents")
            iteration_count += 1

        return current_results

    def _get_optimized_queries(self, improved_query: str, query_analysis: RetrieverQueryAnalysis):
        try:
            return [
                *self._people_queries(improved_query, query_analysis),
                *self._team_queries(improved_query, query_analysis),
                *self._job_queries(improved_query, query_analysis),
                *self._interview_queries(improved_query, query_analysis),
            ]
        except Exception as e:
            logger.error(f"Error generating optimized queries: {e}")
            return []

    def _get_initial_results(self, optimized_queries: List[RetrieverOptimizedQuery], query_analysis: RetrieverQueryAnalysis, top_k):
        results = []
        for optimized_query in optimized_queries:
            query_results = self._get_query_results(
                optimized_query.optimized_query,
                query_analysis,
                top_k
            )
            results.extend(query_results)
        return results

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
        # Create a lookup dictionary for efficient access
        if not results:
            return []

        result_lookup = {result.doc_id: result for result in results}

        # Organize results by base ID
        base_id_groups = {}
        for result in results:
            dedup_id = self._get_base_id(result)
            if dedup_id not in base_id_groups:
                base_id_groups[dedup_id] = []
            base_id_groups[dedup_id].append(result)

        # Select highest similarity document from each group
        unique_results = []
        for group in base_id_groups.values():
            unique_results.append(max(group, key=lambda x: x.similarity))

        # Score the unique results
        scored_results = self.result_analysis_service.score_results(
            improved_query, query_analysis, unique_results)

        # Apply threshold but ensure diversity if possible
        selected_results = []
        doc_types_included = set()

        # First pass: add high-relevance documents
        for scored_doc in sorted(scored_results.documents, key=lambda x: x.relevance_score, reverse=True):
            if scored_doc.relevance_score >= self.relevance_threshold:
                original_doc = result_lookup[scored_doc.doc_id]
                selected_results.append(original_doc)
                doc_type = original_doc.metadata.get(
                    "document_type", "unknown")
                doc_types_included.add(doc_type)

        # Second pass: ensure diversity by adding at least one document of each type
        if query_analysis.information_type:
            for info_type in query_analysis.information_type:
                info_type_value = info_type.value
                if info_type_value not in doc_types_included:
                    # Find highest scoring document of this type
                    for scored_doc in sorted(scored_results.documents, key=lambda x: x.relevance_score, reverse=True):
                        original_doc = result_lookup[scored_doc.doc_id]
                        doc_type = original_doc.metadata.get(
                            "document_type", "unknown")
                        if doc_type == info_type_value and original_doc not in selected_results:
                            selected_results.append(original_doc)
                            doc_types_included.add(doc_type)
                            break

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
        query_complexity = self.query_complexity_service.estimate_query_complexity(
            query, query_analysis)
        query_hierarchy_level = self.query_complexity_service.select_hierarchy_level(
            query_complexity)

        logger.info(
            f"Using hierarchy level '{query_hierarchy_level}' and filter criteria '{filter_criteria}' for query: {query}")

        return self.vector_db.search(
            query,
            top_k=top_k * 2,
            hierarchy_preference=query_hierarchy_level,
            filter_criteria=filter_criteria,
        )

    def _perform_iteration(self, improved_query: str, query_analysis: RetrieverQueryAnalysis, results: List[DocumentResults]):
        """
        Performs an iteration of retrieval to find additional relevant information.

        Args:
            improved_query: The improved query text
            query_analysis: The query analysis object
            selected_results: Currently selected results

        Returns:
            List of additional relevant documents to add to results
        """
        if not results:
            return []

        formatted_results = self.result_analysis_service._format_results_for_analysis(
            results)

        analysis_result = self.result_analysis_service.analyze_results(
            improved_query, query_analysis, formatted_results)

        if analysis_result.status == ResultAnswer.COMPLETE:
            logger.info(
                f"Results are complete. Analysis: {analysis_result.explanation}")
            return []

        logger.info(
            f"Results are incomplete. Generating follow-up queries. Gaps: {', '.join([x.information_type.name for x in analysis_result.information_gaps])}")

        information_gaps = {
            x.information_type for x in analysis_result.information_gaps}
        follow_up_queries = []

        type_to_func = {
            InformationGapType.JOB_OPENING: self.query_analysis_service.generate_job_queries,
            InformationGapType.TEAM_STRUCTURE: self.query_analysis_service.generate_team_queries,
            InformationGapType.RESUME: self.query_analysis_service.generate_person_queries,
            InformationGapType.MEETING_NOTES: self.query_analysis_service.generate_interview_queries,
        }
        gap_to_document_type = {
            InformationGapType.JOB_OPENING: DocumentType.JOB_OPENING,
            InformationGapType.TEAM_STRUCTURE: DocumentType.TEAM_STRUCTURE,
            InformationGapType.RESUME: DocumentType.RESUME,
            InformationGapType.MEETING_NOTES: DocumentType.MEETING_NOTES,
        }

        for information_gap in information_gaps:
            query_result = type_to_func[information_gap](
                improved_query, query_analysis)
            if query_result.optimized_queries:
                follow_up_queries.append({
                    "queries": query_result.optimized_queries,
                    "filter_criteria": {
                        "document_type": gap_to_document_type[information_gap].value
                    }
                })

        if not follow_up_queries:
            logger.info("No follow-up queries generated.")
            return []

        seen_doc_ids = {self._get_base_id(result)
                        for result in results}

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
                seen_doc_ids.update(self._get_base_id(result)
                                    for result in new_results)

        if additional_results:
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
        enhanced_results = []
        added_ids = set()

        for result in results:
            if result.doc_id in added_ids:
                continue

            if result.metadata.get("hierarchy_level") == "document":
                reconstructed_text = self._reconstruct_document_from_sections(
                    result)
                if reconstructed_text:
                    result = DocumentResults(
                        doc_id=result.doc_id,
                        similarity=result.similarity,
                        text=reconstructed_text,
                        metadata=result.metadata
                    )

                enhanced_results.append(result)
                added_ids.add(result.doc_id)

                if result.metadata.get("section_chunks"):
                    for section_id in result.metadata.get("section_chunks").split(","):
                        added_ids.add(section_id)

                if result.metadata.get("base_chunks"):
                    for base_id in result.metadata.get("base_chunks").split(","):
                        added_ids.add(base_id)

            elif result.metadata.get("hierarchy_level") == "section":
                enhanced_results.append(result)
                added_ids.add(result.doc_id)

                if result.metadata.get("base_chunks"):
                    for base_id in result.metadata.get("base_chunks").split(","):
                        added_ids.add(base_id)

            elif result.metadata.get("hierarchy_level") == "base":
                enhanced_results.append(result)
                added_ids.add(result.doc_id)

                self._add_adjacent_chunks(result, enhanced_results, added_ids)

                self._add_parent_section(result, enhanced_results, added_ids)

        enhanced_results.sort(key=lambda x: x.similarity, reverse=True)

        return enhanced_results

    def _reconstruct_document_from_sections(self, doc_result):
        if not doc_result.metadata.get("section_chunks"):
            return None

        section_ids = doc_result.metadata.get("section_chunks").split(",")

        section_chunks = []
        for section_id in section_ids:
            section_summary = self.vector_db.get_document_summary(section_id)
            if section_summary and section_summary.get("summary_text"):
                section_chunks.append(section_summary["summary_text"])

        if section_chunks:
            return "\n\n".join(section_chunks)

        if doc_result.metadata.get("base_chunks"):
            base_ids = doc_result.metadata.get("base_chunks").split(",")
            base_chunks = []
            for base_id in base_ids:
                base_summary = self.vector_db.get_document_summary(base_id)
                if base_summary and base_summary.get("summary_text"):
                    base_chunks.append(base_summary["summary_text"])

            if base_chunks:
                return "\n\n".join(base_chunks)

        return None

    def _add_adjacent_chunks(self, result, enhanced_results, added_ids):
        adj_ids = []

        # Check for next and previous chunks
        if result.metadata.get("next_chunk_id") and result.metadata.get("next_chunk_id") not in added_ids:
            adj_ids.append(result.metadata["next_chunk_id"])

        if result.metadata.get("prev_chunk_id") and result.metadata.get("prev_chunk_id") not in added_ids:
            adj_ids.append(result.metadata["prev_chunk_id"])

        if not adj_ids:
            return

        # Retrieve adjacent chunks using get_document_summary
        for adj_id in adj_ids:
            if adj_id not in added_ids:
                adj_summary = self.vector_db.get_document_summary(adj_id)
                if adj_summary and adj_summary.get("summary_text"):
                    # Add with slightly lower similarity to indicate contextual relationship
                    enhanced_results.append(DocumentResults(
                        doc_id=adj_id,
                        similarity=result.similarity * 0.85,
                        text=adj_summary["summary_text"],
                        metadata=adj_summary.get("metadata", {})
                    ))
                    added_ids.add(adj_id)

    def _add_parent_section(self, result, enhanced_results, added_ids):
        """Helper method to add parent section for context"""
        doc_id = result.metadata.get("parent_id", "")
        if not doc_id:
            return

        # Find section containing this chunk using get_similar_documents
        similar_docs = self.vector_db.get_similar_documents(
            text=result.text,
            top_k=1,
            filter_criteria={
                "hierarchy_level": "section",
                "parent_id": doc_id,
                "base_chunks": {"$in": [result.doc_id]}
            }
        )

        if similar_docs:
            section_doc = similar_docs[0]
            section_id = section_doc["matched_chunk"]["chunk_id"]
            if section_id not in added_ids:
                enhanced_results.append(DocumentResults(
                    doc_id=section_id,
                    similarity=result.similarity * 0.9,
                    text=section_doc["matched_chunk"]["text"],
                    metadata=section_doc["matched_chunk"]
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
