import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
from dataclasses import asdict
import re

from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random_exponential

from app.schemas.document_results import DocumentResults

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorDatabase:
    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "text-embedding-3-small",
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
        base_chunk_size: int = 500,
        chunk_overlap_ratio: float = 0.1,
        enable_telemetry: bool = True
    ):
        """Initialize the Vector Database with OpenAI embeddings and ChromaDB

        Args:
            openai_api_key: API key for OpenAI
            model_name: Embedding model to use
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
            base_chunk_size: Base chunk size to use for hierarchical chunking
            chunk_overlap_ratio: Ratio of chunk size to use for overlap
            enable_telemetry: Whether to log performance metrics
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.base_chunk_size = base_chunk_size
        self.chunk_overlap_ratio = chunk_overlap_ratio
        self.enable_telemetry = enable_telemetry

        self.openai_client = OpenAI(api_key=openai_api_key)

        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=model_name
        )

        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Document collection for RAG"}
            )
            logger.info(
                f"Initialized ChromaDB collection '{collection_name}' at '{persist_directory}'")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_document(
        self,
        doc_id: Union[str, int],
        document_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        base_chunk_size: Optional[int] = None,
        paragraph_break_chars: str = "\n\n",
        section_break_markers: List[str] = [
            "# ", "## ", "### ", "SECTION:", "TITLE:"]
    ) -> Dict[str, List[str]]:
        """Add a document with hierarchical chunking using natural boundaries

        Args:
            doc_id: Unique identifier for the document
            document_text: Text content of the document
            metadata: Additional metadata for the document
            base_chunk_size: Target size for base chunks (not a hard limit)
            paragraph_break_chars: Characters indicating paragraph breaks
            section_break_markers: Text markers indicating section breaks

        Returns:
            Dictionary mapping hierarchy levels to lists of chunk IDs
        """
        start_time = time.time()
        metadata = metadata or {}
        base_chunk_size = base_chunk_size or self.base_chunk_size

        # 1. Identify document structure and natural breaks
        structure = self._analyze_document_structure(
            document_text,
            paragraph_breaks=paragraph_break_chars,
            section_markers=section_break_markers
        )

        # 2. Create base chunks respecting natural boundaries
        base_chunks = self._create_dynamic_chunks(
            document_text,
            structure,
            target_size=base_chunk_size
        )

        # 3. Create hierarchical layers
        hierarchy = self._build_chunk_hierarchy(
            base_chunks, structure, doc_id, document_text)

        # 4. Store all chunks with their hierarchical metadata
        chunk_ids_by_level = self._store_hierarchical_chunks(
            hierarchy, doc_id, metadata
        )

        # 5. Create document summary/metadata node
        self._create_hierarchy_metadata(
            doc_id, metadata, hierarchy, chunk_ids_by_level
        )

        if self.enable_telemetry:
            duration = time.time() - start_time
            logger.info(
                f"Added document {doc_id} with hierarchical chunking in {duration:.4f}s. " +
                f"Created {len(chunk_ids_by_level['base'])} base chunks, " +
                f"{len(chunk_ids_by_level['section'])} section chunks, " +
                f"{len(chunk_ids_by_level['document'])} document chunks."
            )

        return chunk_ids_by_level

    def _analyze_document_structure(
        self,
        document_text: str,
        paragraph_breaks: str = "\n\n",
        section_markers: List[str] = [
            "# ", "## ", "### ", "SECTION:", "TITLE:"]
    ) -> Dict[str, Any]:
        """Analyze document structure to identify natural boundaries

        Returns:
            Dictionary with document structure information
        """
        # Split text by paragraphs
        paragraphs = document_text.split(paragraph_breaks)

        # Identify section boundaries
        sections = []
        current_section = {"title": "Introduction",
                           "start": 0, "paragraphs": []}

        for i, para in enumerate(paragraphs):
            # Check if paragraph starts with a section marker
            is_section_start = any(para.strip().startswith(marker)
                                   for marker in section_markers)

            if is_section_start and i > 0:
                # Complete previous section
                current_section["end"] = i - 1
                sections.append(current_section)
                # Start new section
                current_section = {
                    "title": para.strip(),
                    "start": i,
                    "paragraphs": []
                }

            current_section["paragraphs"].append(para)

        # Add final section
        current_section["end"] = len(paragraphs) - 1
        sections.append(current_section)

        # Calculate paragraph lengths (approximate tokens)
        # Rough approximation, 4 chars ≈ 1 token
        para_lengths = [len(p) // 4 for p in paragraphs]

        return {
            "paragraphs": paragraphs,
            "paragraph_lengths": para_lengths,
            "sections": sections,
            "total_paragraphs": len(paragraphs)
        }

    def _create_dynamic_chunks(
        self,
        document_text: str,
        structure: Dict[str, Any],
        target_size: int = 500
    ) -> List[Dict[str, Any]]:
        """Create chunks dynamically based on document structure

        Args:
            document_text: Full document text
            structure: Document structure from analysis
            target_size: Target chunk size in tokens (approximate)

        Returns:
            List of chunk information dictionaries
        """
        chunks = []
        paragraphs = structure["paragraphs"]
        para_lengths = structure["paragraph_lengths"]

        current_chunk = {
            "paragraphs": [],
            "start_para_idx": 0,
            "text": "",
            "token_count": 0
        }

        for i, (para, length) in enumerate(zip(paragraphs, para_lengths)):
            # Check if adding this paragraph would exceed target size
            if current_chunk["token_count"] + length > target_size and current_chunk["paragraphs"]:
                # Complete current chunk if it has paragraphs
                current_chunk["end_para_idx"] = i - 1
                chunks.append(current_chunk)

                # Start new chunk
                current_chunk = {
                    "paragraphs": [para],
                    "start_para_idx": i,
                    "text": para,
                    "token_count": length
                }
            else:
                # Add paragraph to current chunk
                current_chunk["paragraphs"].append(para)
                current_chunk["text"] += ("\n\n" if current_chunk["text"]
                                          else "") + para
                current_chunk["token_count"] += length

        # Add final chunk if not empty
        if current_chunk["paragraphs"]:
            current_chunk["end_para_idx"] = len(paragraphs) - 1
            chunks.append(current_chunk)

        # Enrich chunks with section information
        for chunk in chunks:
            # Find the section(s) this chunk belongs to
            chunk_sections = []
            for section in structure["sections"]:
                if (chunk["start_para_idx"] <= section["end"] and
                        chunk["end_para_idx"] >= section["start"]):
                    chunk_sections.append(section["title"])

            chunk["sections"] = chunk_sections

        return chunks

    def _build_chunk_hierarchy(
        self,
        base_chunks: List[Dict[str, Any]],
        structure: Dict[str, Any],
        doc_id: Union[str, int],
        document_text: str
    ) -> Dict[str, Any]:
        """Build hierarchy of chunks

        Args:
            base_chunks: Base level chunks
            structure: Document structure
            doc_id: Document ID
            document_text: Full document text

        Returns:
            Hierarchical structure of chunks
        """
        # Level 0: Base chunks (already created)

        # Level 1: Section level chunks
        section_chunks = []
        for section in structure["sections"]:
            # Find all base chunks that belong to this section
            section_base_chunks = []
            for i, chunk in enumerate(base_chunks):
                if any(sect in chunk["sections"] for sect in [section["title"]]):
                    section_base_chunks.append(i)

            if section_base_chunks:
                # Create section chunk
                section_text = "\n\n".join(
                    [base_chunks[i]["text"] for i in section_base_chunks])
                section_chunk = {
                    "text": section_text,
                    "title": section["title"],
                    "base_chunks": section_base_chunks,
                    "token_count": sum(base_chunks[i]["token_count"] for i in section_base_chunks)
                }
                section_chunks.append(section_chunk)

        # Level 2: Document level chunk
        document_chunk = {
            "text": document_text,
            "title": f"Complete document {doc_id}",
            "section_chunks": list(range(len(section_chunks))),
            "base_chunks": list(range(len(base_chunks))),
            "token_count": sum(c["token_count"] for c in base_chunks)
        }

        return {
            "base": base_chunks,
            "section": section_chunks,
            "document": document_chunk
        }

    def _store_hierarchical_chunks(
        self,
        hierarchy: Dict[str, Any],
        doc_id: Union[str, int],
        base_metadata: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Store all hierarchical chunks in the vector database

        Args:
            hierarchy: Hierarchical chunk structure
            doc_id: Document ID
            base_metadata: Base metadata to include

        Returns:
            Dictionary mapping levels to lists of chunk IDs
        """
        chunk_ids = {"base": [], "section": [], "document": []}

        # Store base chunks
        for i, chunk in enumerate(hierarchy["base"]):
            chunk_id = f"{doc_id}_base_{i}"

            # Create metadata
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "hierarchy_level": "base",
                "hierarchy_index": i,
                "parent_id": str(doc_id),
                "token_count": chunk["token_count"],
                "paragraph_range": f"{chunk['start_para_idx']}-{chunk['end_para_idx']}",
                "sections": ",".join(chunk["sections"]),
                "next_chunk_id": f"{doc_id}_base_{i+1}" if i < len(hierarchy["base"])-1 else "",
                "prev_chunk_id": f"{doc_id}_base_{i-1}" if i > 0 else ""
            })

            # Store in vector DB
            self.collection.add(
                documents=[chunk["text"]],
                ids=[chunk_id],
                metadatas=[chunk_metadata]
            )

            chunk_ids["base"].append(chunk_id)

        # Store section chunks
        for i, chunk in enumerate(hierarchy["section"]):
            chunk_id = f"{doc_id}_section_{i}"

            # Create metadata
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "hierarchy_level": "section",
                "hierarchy_index": i,
                "parent_id": str(doc_id),
                "token_count": chunk["token_count"],
                "title": chunk["title"],
                "base_chunks": ",".join([chunk_ids["base"][j] for j in chunk["base_chunks"]]),
                "next_section_id": f"{doc_id}_section_{i+1}" if i < len(hierarchy["section"])-1 else "",
                "prev_section_id": f"{doc_id}_section_{i-1}" if i > 0 else ""
            })

            # Store in vector DB
            self.collection.add(
                documents=[chunk["text"]],
                ids=[chunk_id],
                metadatas=[chunk_metadata]
            )

            chunk_ids["section"].append(chunk_id)

        # Store document chunk
        doc_chunk = hierarchy["document"]
        doc_chunk_id = f"{doc_id}_document"

        # Create metadata
        doc_metadata = base_metadata.copy()
        doc_metadata.update({
            "hierarchy_level": "document",
            "parent_id": str(doc_id),
            "token_count": doc_chunk["token_count"],
            "base_chunks": ",".join(chunk_ids["base"]),
            "section_chunks": ",".join(chunk_ids["section"])
        })

        # This full document might exceed token limits, so store metadata only
        # and only store first ~1000 characters of the document as a preview
        preview_text = doc_chunk["text"][:1000] + \
            "..." if len(doc_chunk["text"]) > 1000 else doc_chunk["text"]
        self.collection.add(
            documents=[
                f"Document summary for {doc_id}. Full hierarchical document."],
            ids=[doc_chunk_id],
            metadatas=[doc_metadata]
        )

        chunk_ids["document"] = [doc_chunk_id]

        return chunk_ids

    def _create_hierarchy_metadata(
        self,
        doc_id: Union[str, int],
        metadata: Dict[str, Any],
        hierarchy: Dict[str, Any],
        chunk_ids: Dict[str, List[str]]
    ) -> None:
        """Create metadata record for the hierarchical document

        Args:
            doc_id: Document ID
            metadata: Original document metadata
            hierarchy: Hierarchy structure
            chunk_ids: Mapping of hierarchy levels to chunk IDs
        """
        hierarchy_metadata = metadata.copy()
        hierarchy_metadata.update({
            "is_hierarchical": True,
            "base_chunks": len(chunk_ids["base"]),
            "section_chunks": len(chunk_ids["section"]),
            "structure_sections": len(hierarchy["section"]),
            "total_chunks": len(chunk_ids["base"]) + len(chunk_ids["section"]) + len(chunk_ids["document"]),
            "chunking_strategy": "hierarchical_dynamic"
        })

        self.collection.add(
            documents=[f"Hierarchical document metadata for {doc_id}"],
            ids=[f"{doc_id}_hierarchy_metadata"],
            metadatas=[hierarchy_metadata]
        )

    def _add_document_with_chunking(
        self,
        doc_id: Union[str, int],
        document_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_sizes: Optional[List[int]] = None,
        chunk_overlap_ratio: Optional[float] = None
    ) -> Dict[str, List[str]]:
        """Add a document with legacy multi-size chunking

        Args:
            doc_id: Unique identifier for the document
            document_text: Text content of the document
            metadata: Additional metadata for the document
            chunk_sizes: List of chunk sizes to use (defaults to [500, 1000, 2000])
            chunk_overlap_ratio: Ratio of chunk size to use for overlap

        Returns:
            Dictionary of chunk types to lists of chunk IDs
        """
        start_time = time.time()
        metadata = metadata or {}
        chunk_sizes = chunk_sizes or [500, 1000, 2000]
        chunk_overlap_ratio = chunk_overlap_ratio or self.chunk_overlap_ratio

        all_chunk_ids = {}

        for size in chunk_sizes:
            overlap = int(size * chunk_overlap_ratio)

            size_metadata = metadata.copy()
            size_metadata.update({
                "chunk_type": f"size_{size}",
                "chunk_size": size,
                "chunk_overlap": overlap,
                "chunking_strategy": "legacy_multi_size"
            })

            chunk_type = f"size_{size}"
            size_doc_id = f"{doc_id}_{chunk_type}"

            # Add document with this specific chunk size
            size_chunk_ids = self._add_document_with_chunking_size(
                size_doc_id,
                document_text,
                size_metadata,
                chunk_size=size,
                chunk_overlap=overlap
            )

            all_chunk_ids[chunk_type] = size_chunk_ids

            if self.enable_telemetry:
                logger.info(
                    f"Added {len(size_chunk_ids)} chunks of size {size} for document {doc_id}")

        summary_metadata = metadata.copy()
        summary_metadata.update({
            "is_document_summary": True,
            "chunk_sizes_used": ", ".join(map(str, chunk_sizes)),
            "total_chunks": sum(len(ids) for ids in all_chunk_ids.values()),
            "chunking_strategy": "legacy_multi_size"
        })

        self.collection.add(
            documents=[
                f"Document summary for {doc_id}. Contains multiple chunk sizes: {chunk_sizes}"],
            ids=[f"{doc_id}_summary"],
            metadatas=[summary_metadata]
        )

        if self.enable_telemetry:
            duration = time.time() - start_time
            logger.info(
                f"Completed legacy chunking for document {doc_id} in {duration:.4f}s")

        return all_chunk_ids

    def _add_document_with_chunking_size(
        self,
        doc_id: Union[str, int],
        document_text: str,
        metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Add a document split into chunks of a specific size

        Args:
            doc_id: Unique identifier for the document
            document_text: Text content of the document
            metadata: Additional metadata for the document
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks

        Returns:
            List of chunk IDs
        """
        chunks = self._chunk_text(document_text, chunk_size, chunk_overlap)

        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        chunk_metadatas = []
        for i in range(len(chunks)):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "parent_id": str(doc_id),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            })
            chunk_metadatas.append(chunk_metadata)

        self.add_documents_batch(chunk_ids, chunks, chunk_metadatas)

        return chunk_ids

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks

        Args:
            text: Text to split
            chunk_size: Size of chunks
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)

            # Try to find a natural break point (newline or period)
            if end < text_len and end > start + (chunk_size // 2):
                # Look for newline
                newline_pos = text.rfind('\n', start, end)
                if newline_pos > start + (chunk_size // 4):
                    end = newline_pos + 1
                else:
                    # Look for period followed by space
                    period_pos = text.rfind('. ', start, end)
                    if period_pos > start + (chunk_size // 4):
                        end = period_pos + 2

            chunks.append(text[start:end])
            start += (chunk_size - overlap)

        return chunks

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=4, max=10))
    def add_documents_batch(
        self,
        doc_ids: List[Union[str, int]],
        document_texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add multiple documents in a single batch operation

        Args:
            doc_ids: List of document IDs
            document_texts: List of document texts
            metadatas: List of metadata dictionaries
        """
        start_time = time.time()

        if metadatas is None:
            metadatas = [{} for _ in doc_ids]

        try:
            self.collection.add(
                documents=document_texts,
                ids=[str(doc_id) for doc_id in doc_ids],
                metadatas=metadatas
            )

            if self.enable_telemetry:
                duration = time.time() - start_time
                logger.info(
                    f"Added {len(doc_ids)} documents in batch in {duration:.4f}s")
        except Exception as e:
            logger.error(f"Error adding documents batch: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=4, max=10))
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
        hierarchy_preference: str = "auto",
        _internal_recursion_depth: int = 0  # Add recursion depth tracker
    ) -> List[DocumentResults]:
        """Search with hierarchy-aware retrieval

        Args:
            query: Search query
            top_k: Number of results to return
            filter_criteria: Filter to apply
            hierarchy_preference: Which level to prefer ("base", "section", "document", or "auto")
            _internal_recursion_depth: Internal parameter to prevent infinite recursion

        Returns:
            List of DocumentResults
        """
        # Prevent recursion depth issues
        if _internal_recursion_depth > 2:
            logger.warning(
                f"Search recursion depth exceeded, returning empty results for query: '{query[:30]}...'")
            return []

        start_time = time.time()
        logger.info(
            f"Beginning hierarchical search with query: '{query[:30]}...' and level: {hierarchy_preference}")

        try:
            # Determine query complexity to choose appropriate hierarchy level
            if hierarchy_preference == "auto":
                query_complexity = self._estimate_query_complexity(query)

                if query_complexity < 0.3:  # Simple, specific query
                    hierarchy_level = "base"
                    search_k = top_k * 3  # Retrieve more base chunks
                elif query_complexity < 0.7:  # Medium complexity
                    hierarchy_level = "section"
                    search_k = top_k * 2
                else:  # Complex query needing broader context
                    hierarchy_level = "document"
                    search_k = top_k

                logger.info(
                    f"Auto-selected hierarchy level: {hierarchy_level} for query complexity {query_complexity:.2f}")
            else:
                hierarchy_level = hierarchy_preference
                search_k = top_k * 2 if hierarchy_level == "base" else top_k

            # Create hierarchy-specific filter
            search_filter = filter_criteria.copy() if filter_criteria else {}
            if hierarchy_level != "all":
                search_filter["hierarchy_level"] = hierarchy_level

            logger.info(f"Using search filter: {search_filter}")

            # IMPORTANT: Use direct ChromaDB query instead of calling self.search() to avoid recursion
            search_params = self._build_search_params(query, search_k, search_filter)

            # Directly query the collection
            raw_results = self.collection.query(**search_params)

            # Format the results manually
            results = []
            if raw_results and raw_results.get('ids') and raw_results['ids'][0]:
                for i in range(len(raw_results['ids'][0])):
                    results.append(DocumentResults(
                        doc_id=raw_results['ids'][0][i],
                        similarity=1.0 -
                        raw_results['distances'][0][i] if 'distances' in raw_results and raw_results['distances'][0] else 0.0,
                        text=raw_results['documents'][0][i],
                        metadata=raw_results['metadatas'][0][i] if raw_results.get(
                            'metadatas') and raw_results['metadatas'][0] else {}
                    ))

            logger.info(
                f"Retrieved {len(results)} initial results at level '{hierarchy_level}'")

            # Process results based on hierarchy
            if not results:
                # Fall back to searching all levels if no results found
                if hierarchy_level != "all":
                    logger.info(
                        f"No results found at level '{hierarchy_level}', falling back to all levels")
                    search_filter = filter_criteria.copy() if filter_criteria else {}

                    search_params = self._build_search_params(query, top_k, search_filter)

                    raw_results = self.collection.query(**search_params)

                    # Format the results manually again
                    if raw_results and raw_results.get('ids') and raw_results['ids'][0]:
                        for i in range(len(raw_results['ids'][0])):
                            results.append(DocumentResults(
                                doc_id=raw_results['ids'][0][i],
                                similarity=1.0 -
                                raw_results['distances'][0][i] if 'distances' in raw_results and raw_results['distances'][0] else 0.0,
                                text=raw_results['documents'][0][i],
                                metadata=raw_results['metadatas'][0][i] if raw_results.get(
                                    'metadatas') and raw_results['metadatas'][0] else {}
                            ))

                    logger.info(
                        f"Retrieved {len(results)} results after fallback to all levels")

            # If we have section or document results, retrieve related base chunks as needed
            if hierarchy_level in ["section", "document"] and results:
                # Extract base chunk IDs from results
                related_base_chunks = set()
                for result in results:
                    if "base_chunks" in result.metadata:
                        chunk_ids = result.metadata["base_chunks"].split(",")
                        related_base_chunks.update(chunk_ids)

                # If we have related base chunks, retrieve them to provide detail
                if related_base_chunks and len(results) < top_k:
                    additional_count = min(
                        top_k - len(results), len(related_base_chunks))
                    logger.info(
                        f"Retrieving {additional_count} related base chunks for context")

                    additional_results = self.collection.get(
                        ids=list(related_base_chunks)[:additional_count]
                    )

                    # Format and add to results
                    for i in range(len(additional_results["ids"])):
                        results.append(DocumentResults(
                            doc_id=additional_results["ids"][i],
                            similarity=0.7,  # Lower similarity as these are related, not direct matches
                            text=additional_results["documents"][i],
                            metadata=additional_results["metadatas"][i]
                        ))

                    logger.info(
                        f"Added {len(additional_results['ids'])} related base chunks")

            if self.enable_telemetry:
                duration = time.time() - start_time
                logger.info(
                    f"Hierarchical search at level '{hierarchy_level}' completed in {duration:.4f}s with {len(results)} results"
                )

            return results[:top_k]
        except Exception as e:
            # Add exc_info for stack trace
            logger.error(
                f"Error during hierarchical search: {e}", exc_info=True)
            # Return empty list instead of raising to prevent retry loops
            return []
    
    def _build_search_params(self, query: str, top_k: int, filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build search parameters for the ChromaDB query

        Args:
            query: Search query
            top_k: Number of results to return
            filter_criteria: Filter criteria for the query

        Returns:
            Dictionary with search parameters
        """
        search_params = {
            "query_texts": [query],
            "n_results": top_k,
        }

        if filter_criteria:
            if len(filter_criteria) > 1:
                where_conditions = []
                for key, value in filter_criteria.items():
                    where_conditions.append({key: value})
                search_params["where"] = {"$and": where_conditions}
            else:
                search_params["where"] = filter_criteria

        return search_params
    
    def _estimate_query_complexity(self, query: str) -> float:
        """Estimate the complexity of a query to determine appropriate hierarchy level

        Args:
            query: Search query

        Returns:
            Complexity score between 0-1
        """
        # Simple heuristics:
        # 1. Query length (longer queries tend to be more complex)
        length_factor = min(len(query) / 100, 1.0) * 0.3

        # 2. Number of specific entities (more entities = more specific query)
        # Simple named entity detection
        entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', query)
        entity_count = len(entities)
        specificity_factor = (
            1.0 - min(entity_count / 5, 1.0)) * 0.3  # Inverse relation

        # 3. Question complexity based on question words
        complex_markers = ['why', 'how', 'explain',
                           'compare', 'analyze', 'summarize', 'relationship']
        simple_markers = ['what', 'who', 'when', 'where', 'list', 'find']

        query_lower = query.lower()
        complex_count = sum(
            1 for word in complex_markers if word in query_lower)
        simple_count = sum(1 for word in simple_markers if word in query_lower)

        question_complexity = 0.5  # Default middle value
        if complex_count > 0 or simple_count > 0:
            total = complex_count + simple_count
            question_complexity = complex_count / total if total > 0 else 0.5

        question_factor = question_complexity * 0.4

        # Combine factors
        complexity = length_factor + specificity_factor + question_factor
        return min(max(complexity, 0.0), 1.0)  # Ensure between 0-1

    def get_document_summary(self, doc_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Get summary information for a document including its chunk distribution

        Args:
            doc_id: Document ID

        Returns:
            Dictionary with document summary information or None if not found
        """
        try:
            # Try to get the hierarchy metadata
            results = self.collection.get(
                ids=[f"{doc_id}_hierarchy_metadata"]
            )

            if results and results["ids"]:
                # Return hierarchy info
                return {
                    "document_id": doc_id,
                    "metadata": results["metadatas"][0],
                    "is_hierarchical": True,
                    "summary_text": results["documents"][0]
                }

            # If no dedicated hierarchy metadata, try to get the legacy summary
            results = self.collection.get(
                ids=[f"{doc_id}_summary"]
            )

            if results and results["ids"]:
                # Return summary info
                return {
                    "document_id": doc_id,
                    "metadata": results["metadatas"][0],
                    "is_hierarchical": False,
                    "summary_text": results["documents"][0]
                }

            # If no dedicated summary, try to get any chunks
            # First try hierarchical
            base_results = self.collection.get(
                where={"parent_id": str(doc_id), "hierarchy_level": "base"},
                limit=1
            )

            if base_results and base_results["ids"]:
                # Count chunks at each level
                base_count = len(self.collection.get(
                    where={"parent_id": str(
                        doc_id), "hierarchy_level": "base"},
                    limit=1000
                )["ids"])

                section_count = len(self.collection.get(
                    where={"parent_id": str(
                        doc_id), "hierarchy_level": "section"},
                    limit=100
                )["ids"])

                return {
                    "document_id": doc_id,
                    "is_hierarchical": True,
                    "base_chunks": base_count,
                    "section_chunks": section_count,
                    "has_dedicated_summary": False
                }

            # Then try legacy
            chunk_results = self.collection.get(
                where={"parent_id": str(doc_id)},
                limit=1
            )

            if chunk_results and chunk_results["ids"]:
                # Get all chunk sizes for this document
                chunk_sizes = set()
                chunk_counts = {}

                # Query for chunks with different chunk sizes
                for size in [500, 1000, 2000]:  # Default sizes in legacy approach
                    size_chunks = self.collection.get(
                        where={"parent_id": str(doc_id), "chunk_size": size},
                        limit=1000  # Set a reasonable limit
                    )

                    if size_chunks and size_chunks["ids"]:
                        chunk_sizes.add(size)
                        chunk_counts[size] = len(size_chunks["ids"])

                # Create a summary from found chunks
                return {
                    "document_id": doc_id,
                    "chunk_sizes": list(chunk_sizes),
                    "chunk_counts": chunk_counts,
                    "total_chunks": sum(chunk_counts.values()),
                    "has_dedicated_summary": False,
                    "is_hierarchical": False
                }

            return None
        except Exception as e:
            logger.error(f"Error getting document summary for {doc_id}: {e}")
            return None

    def add_document_version(
        self,
        doc_id: Union[str, int],
        document_text: str,
        version: Union[str, int],
        metadata: Optional[Dict[str, Any]] = None,
        use_hierarchical: bool = True
    ) -> Dict[str, List[str]]:
        """Add a new version of an existing document

        Args:
            doc_id: Base document ID
            document_text: Document text
            version: Version identifier
            metadata: Additional metadata
            use_hierarchical: Whether to use hierarchical chunking

        Returns:
            Dictionary mapping levels to lists of chunk IDs for the new version
        """
        start_time = time.time()

        if metadata is None:
            metadata = {}

        # Add version information to metadata
        version_metadata = metadata.copy()
        version_metadata.update({
            "version": str(version),
            "original_id": str(doc_id),
            "timestamp": time.time(),
            "is_version": True
        })

        # Generate version-specific ID
        version_id = f"{doc_id}_v{version}"

        try:
            # Add versioned document using the appropriate method
            if use_hierarchical:
                chunk_ids = self.add_document_hierarchical(
                    version_id, document_text, version_metadata)
            else:
                chunk_ids = self._add_document_with_chunking(
                    version_id, document_text, version_metadata)

            # Update version reference in original document if it exists
            try:
                # Check if original document exists
                doc_summary = self.get_document_summary(doc_id)

                if doc_summary:
                    # Get appropriate metadata
                    if doc_summary.get("is_hierarchical", False):
                        results = self.collection.get(
                            ids=[f"{doc_id}_hierarchy_metadata"])
                    else:
                        results = self.collection.get(
                            ids=[f"{doc_id}_summary"])

                    if results and results["ids"]:
                        # Update or create version pointer
                        pointer_metadata = results["metadatas"][0] if results["metadatas"] else {
                        }
                        pointer_metadata.update({
                            "latest_version": str(version),
                            "has_versions": True
                        })

                        # Update the pointer document
                        self.collection.update(
                            ids=[results["ids"][0]],
                            metadatas=[pointer_metadata]
                        )
                else:
                    # Create pointer document if it doesn't exist
                    pointer_metadata = {
                        "latest_version": str(version),
                        "has_versions": True,
                        "is_version_pointer": True
                    }

                    # We use a placeholder document text
                    self.collection.add(
                        documents=["[VERSION POINTER - SEE METADATA]"],
                        ids=[str(doc_id)],
                        metadatas=[pointer_metadata]
                    )
            except Exception as e:
                logger.warning(
                    f"Error updating version pointer for document {doc_id}: {e}")

            if self.enable_telemetry:
                duration = time.time() - start_time
                chunking_type = "hierarchical" if use_hierarchical else "legacy multi-size"
                logger.info(
                    f"Added version {version} of document {doc_id} with {chunking_type} chunking in {duration:.4f}s")

            return chunk_ids
        except Exception as e:
            logger.error(
                f"Error adding version {version} of document {doc_id}: {e}")
            raise

    def get_document_versions(self, doc_id: Union[str, int]) -> List[Dict[str, Any]]:
        """Get all versions of a document

        Args:
            doc_id: Document ID

        Returns:
            List of version metadata
        """
        try:
            # First check if the document exists and has versions
            doc_summary = self.get_document_summary(doc_id)

            if not doc_summary:
                logger.warning(f"Document {doc_id} not found")
                return []

            # Get all versions using parent_id
            version_results = self.collection.get(
                where={"original_id": str(doc_id), "is_version": True}
            )

            if not version_results or not version_results["ids"]:
                # Try finding by ID pattern
                version_results = self.collection.get(
                    where_document={"$contains": f"{doc_id}_v"}
                )

            # Format version information
            versions = []
            if version_results and version_results["ids"]:
                for i, vid in enumerate(version_results["ids"]):
                    metadata = version_results["metadatas"][i]
                    versions.append({
                        "version_id": vid,
                        "version": metadata.get("version"),
                        "timestamp": metadata.get("timestamp"),
                        "metadata": metadata,
                        "is_hierarchical": "hierarchy_level" in metadata
                    })

                # Sort by timestamp (newest first)
                versions.sort(key=lambda x: x.get(
                    "timestamp", 0), reverse=True)

            return versions
        except Exception as e:
            logger.error(f"Error getting versions for document {doc_id}: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection

        Returns:
            Dictionary of collection statistics
        """
        try:
            # Get count of documents
            count = self.collection.count()

            # Check for hierarchical documents
            hierarchical_results = self.collection.get(
                where={"hierarchy_level": {"$exists": True}},
                limit=1
            )
            has_hierarchical = hierarchical_results and hierarchical_results["ids"]

            # Check for legacy chunked documents
            chunk_results = self.collection.get(
                where_document={"$contains": "_chunk_"},
                limit=1
            )
            has_legacy_chunks = chunk_results and chunk_results["ids"]

            # Check for versioned documents
            version_results = self.collection.get(
                where={"is_version": True},
                limit=1
            )
            has_versions = version_results and version_results["ids"]

            # Count documents by type
            try:
                hierarchical_count = self.collection.count(
                    where={"is_hierarchical": True})
            except:
                hierarchical_count = 0

            try:
                legacy_count = self.collection.count(
                    where={"chunking_strategy": "legacy_multi_size"})
            except:
                legacy_count = 0

            return {
                "document_count": count,
                "has_hierarchical_documents": has_hierarchical,
                "has_legacy_chunks": has_legacy_chunks,
                "has_versions": has_versions,
                "hierarchical_document_count": hierarchical_count,
                "legacy_document_count": legacy_count,
                "embedding_model": self.model_name,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a text string

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            embeddings = self.embedding_function([text])
            return embeddings[0]
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    def delete_document(self, doc_id: Union[str, int], delete_versions: bool = False) -> bool:
        """Delete a document and all its chunks

        Args:
            doc_id: Document ID to delete
            delete_versions: Whether to also delete all versions of the document

        Returns:
            Success flag
        """
        try:
            # First, determine if it's a hierarchical document
            doc_summary = self.get_document_summary(doc_id)

            if not doc_summary:
                logger.warning(f"Document {doc_id} not found")
                return False

            if doc_summary.get("is_hierarchical", False):
                # Delete hierarchical document pieces
                # 1. Get all associated IDs from metadata
                hierarchy_ids = []

                # Get base chunks
                base_results = self.collection.get(
                    where={"parent_id": str(doc_id), "hierarchy_level": "base"}
                )
                if base_results and base_results["ids"]:
                    hierarchy_ids.extend(base_results["ids"])

                # Get section chunks
                section_results = self.collection.get(
                    where={"parent_id": str(
                        doc_id), "hierarchy_level": "section"}
                )
                if section_results and section_results["ids"]:
                    hierarchy_ids.extend(section_results["ids"])

                # Get document level
                hierarchy_ids.append(f"{doc_id}_document")

                # Get metadata
                hierarchy_ids.append(f"{doc_id}_hierarchy_metadata")

                # Delete all hierarchical pieces
                if hierarchy_ids:
                    self.collection.delete(ids=hierarchy_ids)

            else:
                # Handle legacy document
                # Try to find all chunks with this parent_id
                chunk_results = self.collection.get(
                    where={"parent_id": str(doc_id)}
                )

                if chunk_results and chunk_results["ids"]:
                    self.collection.delete(ids=chunk_results["ids"])

                # Delete summary
                try:
                    self.collection.delete(ids=[f"{doc_id}_summary"])
                except:
                    pass

            # Delete the document pointer itself if it exists
            try:
                self.collection.delete(ids=[str(doc_id)])
            except:
                pass

            # Handle versions if requested
            if delete_versions:
                # Get all versions first
                versions = self.get_document_versions(doc_id)
                for version in versions:
                    version_id = version["version_id"]
                    self.delete_document(version_id, delete_versions=False)

            logger.info(f"Successfully deleted document {doc_id}" +
                        (" and all its versions" if delete_versions else ""))
            return True

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def get_similar_documents(
        self,
        text: str,
        top_k: int = 5,
        use_hierarchical: bool = True,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Find documents similar to the provided text

        Args:
            text: Reference text to match against
            top_k: Number of results to return
            use_hierarchical: Whether to use hierarchical search
            filter_criteria: Additional filtering criteria

        Returns:
            List of similar documents with metadata
        """
        try:
            if use_hierarchical:
                results = self.search_hierarchical(
                    text, top_k=top_k, filter_criteria=filter_criteria)
            else:
                results = self.search(
                    text, top_k=top_k, filter_criteria=filter_criteria)

            # Extract unique parent documents
            unique_docs = {}
            for result in results:
                parent_id = result.metadata.get("parent_id", "unknown")
                if parent_id not in unique_docs:
                    unique_docs[parent_id] = {
                        "doc_id": parent_id,
                        "similarity": result.similarity,
                        "is_hierarchical": "hierarchy_level" in result.metadata,
                        "matched_chunk": {
                            "chunk_id": result.doc_id,
                            "text": result.text,
                            "hierarchy_level": result.metadata.get("hierarchy_level", "unknown")
                        }
                    }

                    # Add document summary if available
                    doc_summary = self.get_document_summary(parent_id)
                    if doc_summary:
                        unique_docs[parent_id]["summary"] = doc_summary

            return list(unique_docs.values())

        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []
