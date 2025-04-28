import re

from app.vector_db import VectorDatabase

class DocumentProcessor:
    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
        
    def process_job_description(self, text, metadata):
        """Process a job description document"""
        # Extract key information
        skills = self._extract_skills(text)
        metadata["extracted_skills"] = ", ".join(skills)
        
        # Store in vector database
        doc_id = f"job_{metadata.get('job_id', 'unknown')}"
        self.vector_db.add_document(doc_id, text, metadata)
        
        return doc_id
        
    def process_resume(self, text, metadata):
        """Process a resume document"""
        # Extract key information
        skills = self._extract_skills(text)
        education = self._extract_education(text)
        
        metadata["extracted_skills"] = ", ".join(skills)
        metadata["extracted_education"] = ", ".join(education)
        
        # Store in vector database
        doc_id = f"resume_{metadata.get('candidate_id', 'unknown')}"
        self.vector_db.add_document(doc_id, text, metadata)
        
        return doc_id
        
    def process_team_structure(self, text, metadata):
        """Process a team structure document"""
        # Store in vector database
        doc_id = f"team_{metadata.get('team_id', 'unknown')}"
        self.vector_db.add_document(doc_id, text, metadata)
        
        return doc_id
        
    def process_meeting_notes(self, text, metadata):
        """Process meeting notes"""
        # Store in vector database
        doc_id = f"meeting_{metadata.get('meeting_id', 'unknown')}"
        self.vector_db.add_document(doc_id, text, metadata)
        
        return doc_id
        
    def _extract_skills(self, text):
        """Simple skill extraction - in a real app, this would be more sophisticated"""
        # List of common tech skills for demo purposes
        common_skills = [
            "python", "javascript", "java", "c\\+\\+", "sql", "react", "node.js", 
            "aws", "azure", "docker", "kubernetes", "machine learning", "data science",
            "agile", "project management", "leadership", "communication"
        ]
        
        found_skills = []
        for skill in common_skills:
            if re.search(r'\b' + skill + r'\b', text.lower()):
                found_skills.append(skill)
                
        return found_skills
        
    def _extract_education(self, text):
        """Simple education extraction - would be more sophisticated in a real app"""
        degrees = ["bachelor", "master", "phd", "mba", "doctorate"]
        found_degrees = []
        
        for degree in degrees:
            if re.search(r'\b' + degree + r'\b', text.lower()):
                found_degrees.append(degree)
                
        return found_degrees