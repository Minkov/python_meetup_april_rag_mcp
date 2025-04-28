from app.mcp import ModelContextProtocol
from app.doc_store import DocumentStore
from app.vector_db import VectorDatabase
from app.document_processor import DocumentProcessor
from app.retriever import Retriever

class HRIntelligenceSystem:
    def __init__(self, openai_api_key: str):
        # Initialize components
        self.document_store = DocumentStore()
        self.vector_db = VectorDatabase(openai_api_key, enable_telemetry=False)
        self.document_processor = DocumentProcessor(self.vector_db)
        
        self.retriever = Retriever(self.vector_db)
        
        self.mcp = ModelContextProtocol()
        
    def add_job_description(self, text, metadata):
        """Add a job description to the system"""
        file_id = self.document_store.store_document("job_descriptions", text, metadata)
        doc_id = self.document_processor.process_job_description(text, metadata)
        return {"file_id": file_id, "doc_id": doc_id}
        
    def add_resume(self, text, metadata):
        """Add a resume to the system"""
        file_id = self.document_store.store_document("resumes", text, metadata)
        doc_id = self.document_processor.process_resume(text, metadata)
        return {"file_id": file_id, "doc_id": doc_id}
        
    def add_team_structure(self, text, metadata):
        """Add a team structure document to the system"""
        file_id = self.document_store.store_document("team_structures", text, metadata)
        doc_id = self.document_processor.process_team_structure(text, metadata)
        return {"file_id": file_id, "doc_id": doc_id}
        
    def add_meeting_notes(self, text, metadata):
        """Add meeting notes to the system"""
        file_id = self.document_store.store_document("meeting_notes", text, metadata)
        doc_id = self.document_processor.process_meeting_notes(text, metadata)
        return {"file_id": file_id, "doc_id": doc_id}
        
    def ask(self, query, top_k=5, max_iterations=5):
        """Ask a question to the HR Intelligence System"""
        retrieval_result = self.retriever.retrieve(
            query, 
            top_k=top_k,
            max_iterations=max_iterations
        )

        retrieved_docs = retrieval_result["retrieved_docs"]
        
        print(f"Original query: {query}")
        print(f"Optimized queries: {retrieval_result['optimized_queries']}")
        print(f"Retrieved {len(retrieved_docs)} documents after {max_iterations} max iterations")

        response = self.mcp.generate_response(query, retrieved_docs)
        
        return {
            "response": response,
            "retrieved_docs": retrieved_docs
        }