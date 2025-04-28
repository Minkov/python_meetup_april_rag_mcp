import os
import json
from datetime import datetime


class DocumentStore:
    def __init__(self, base_dir="hr_documents"):
        self.base_dir = base_dir
        self.document_types = ["job_descriptions",
                               "team_structures", "resumes", "meeting_notes"]
        self._initialize_storage()

    def _initialize_storage(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        for doc_type in self.document_types:
            doc_path = os.path.join(self.base_dir, doc_type)
            if not os.path.exists(doc_path):
                os.makedirs(doc_path)

    def store_document(self, doc_type, document, metadata=None):
        """Store a document with optional metadata"""
        if doc_type not in self.document_types:
            raise ValueError(
                f"Document type must be one of {self.document_types}")

        # Create a unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = f"{timestamp}_{hash(document)}"

        doc_path = os.path.join(self.base_dir, doc_type, f"{file_id}.txt")

        # Store the document
        with open(doc_path, "w") as f:
            f.write(document)

        # Store metadata if provided
        if metadata:
            meta_path = os.path.join(
                self.base_dir, doc_type, f"{file_id}_meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

        return file_id

    def get_document(self, doc_type, file_id):
        """Retrieve a document by type and ID"""
        doc_path = os.path.join(self.base_dir, doc_type, f"{file_id}.txt")

        if not os.path.exists(doc_path):
            return None

        with open(doc_path, "r") as f:
            document = f.read()

        # Try to get metadata if available
        metadata = None
        meta_path = os.path.join(
            self.base_dir, doc_type, f"{file_id}_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        return {"document": document, "metadata": metadata}
