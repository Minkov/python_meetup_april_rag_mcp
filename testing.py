from dataclasses import asdict
import json
import os
from app.schemas.retriever import DocumentType
from app.vector_db import VectorDatabase

vector_db = VectorDatabase(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

results = vector_db.search(
    query='Senior Software Engineer - Public Website Team',
    filter_criteria={"document_type": DocumentType.JOB_OPENING.value, 'level': 'Senior',
                     'hierarchy_level': 'document'},
    top_k=10
)

print(json.dumps(
    [asdict(result) for result in results],
))