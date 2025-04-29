import os
import sys
from pathlib import Path
import re

from dotenv import load_dotenv
from app.chat import Chat
from app.hr_system import HRIntelligenceSystem
from app.schemas.retriever import DocumentType

def read_mock_document(file_path):
    """Read a mock document from the given file path."""
    with open(file_path, 'r') as f:
        return f.read()

def extract_job_metadata(content):
    """Extract metadata from job description content."""
    # Extract title from first line (after #)
    title = content.split('\n')[0].replace('#', '').strip()
    
    # Extract metadata from the next three lines
    lines = content.split('\n')
    position = None
    level = None
    team = None
    
    for line in lines[1:4]:  # Look at lines 2-4
        if line.startswith('- Position:'):
            position = line.replace('- Position:', '').strip()
        elif line.startswith('- Level:'):
            level = line.replace('- Level:', '').strip()
        elif line.startswith('- Team:'):
            team = line.replace('- Team:', '').strip()
    
    # Generate job_id based on title
    job_id = f"job{len(title) % 1000:03d}"
    
    return {
        "job_id": job_id,
        "title": title,
        "team": team,
        "level": level,
        "document_type": DocumentType.JOB_OPENING.value,
        "position": position,
    }

def extract_resume_metadata(content, file_path):
    """Extract metadata from resume filename.
    
    Expected filename format: {FULLNAME}-{CURRENTPOSITION}-{CANDREF}.txt
    Example: john-doe-software-engineer-cand123.txt
    """
    # Get filename without extension
    filename = Path(file_path).stem
    
    # Split filename into parts (assuming format: name-position-candXXX)
    parts = filename.split('-')
    
    # Extract name (first part)
    name = parts[0].replace('_', ' ').title()
    
    # Extract current position (second part)
    current_role = parts[1].replace('_', ' ').title() if len(parts) > 1 else "Unknown"
    
    # Extract candidate_id (third part)
    candidate_id = parts[2] if len(parts) > 2 else f"cand{len(name) % 1000:03d}"
    
    return {
        "candidate_id": candidate_id,
        "candidate_name": name,
        "current_role": current_role,
        "document_type": DocumentType.RESUME.value,
    }

def extract_team_metadata(content):
    """Extract metadata from team structure content."""
    lines = content.split('\n')
    
    # Extract team name from first line (removing # and spaces)
    team_name = lines[0].replace('#', '').strip()
    
    # Extract team lead from the line after team name
    team_lead = None
    for line in lines:
        if '**Team lead**:' in line:
            team_lead = line.split('**Team lead**:')[1].strip()
            break
    
    focus_areas = []
    in_focus_section = False
    for line in lines:
        if '## Focus area:' in line:
            in_focus_section = True
            continue
        if in_focus_section and line.strip().startswith('-'):
            focus_areas.append(line.strip('- '))
        if in_focus_section and line.strip().startswith('##'):
            in_focus_section = False
    
    team_id = f"{team_name.lower().replace(' ', '_')}"
    
    # Convert focus_areas list to a comma-separated string
    focus_areas_str = ', '.join(focus_areas)
    
    return {
        "team_id": team_id,
        "team_name": team_name,
        "team_lead": team_lead,
        "focus_areas": focus_areas_str,
        "document_type": DocumentType.TEAM_STRUCTURE.value,
    }

def extract_meeting_metadata(content):
    """Extract metadata from meeting notes content."""
    # Extract candidate info from first line
    first_line = content.split('\n')[0]
    match = re.search(r'# Interview notes (.*?) \((cand\d+)\)', first_line)
    if match:
        candidate_name = match.group(1)
        candidate_id = match.group(2)
    else:
        candidate_name = "Unknown"
        candidate_id = "cand000"
    
    # Extract date from second line
    date_line = content.split('\n')[1]
    date_match = re.search(r'Date: (.*)', date_line)
    date = date_match.group(1) if date_match else "Unknown"
    
    # Generate meeting_id based on candidate_id
    meeting_id = f"meet{candidate_id[-3:]}"
    
    return {
        "candidate_name": candidate_name,
        "meeting_id": meeting_id,
        "candidate_id": candidate_id,
        "document_type": DocumentType.MEETING_NOTES.value,
        "date": date,
    }

def populate_job_descriptions(hr_system):
    """Add job descriptions to the HR system from mock documents."""
    job_descriptions_dir = Path("mock_documents/job_descriptions")
    
    for file_path in job_descriptions_dir.glob("*.txt"):
        content = read_mock_document(file_path)
        metadata = extract_job_metadata(content)
        hr_system.add_job_description(content, metadata)

def populate_resumes(hr_system):
    """Add resumes to the HR system from mock documents."""
    resumes_dir = Path("mock_documents/resumes")
    
    for file_path in resumes_dir.glob("*.txt"):
        content = read_mock_document(file_path)
        metadata = extract_resume_metadata(content, str(file_path))
        hr_system.add_resume(content, metadata)

def populate_teams(hr_system):
    """Add team structures to the HR system from mock documents."""
    teams_dir = Path("mock_documents/teams")
    
    for file_path in teams_dir.glob("*.txt"):
        content = read_mock_document(file_path)
        metadata = extract_team_metadata(content)
        hr_system.add_team_structure(content, metadata)

def populate_meeting_notes(hr_system):
    """Add meeting notes to the HR system from mock documents."""
    interviews_dir = Path("mock_documents/interviews")
    
    for file_path in interviews_dir.glob("*.txt"):
        content = read_mock_document(file_path)
        metadata = extract_meeting_metadata(content)
        hr_system.add_meeting_notes(content, metadata)

def run_full_demo(populate_data=False):
    """Run the full demo with options for different retrieval and MCP combinations"""
    # Load environment variables (for API key)
    load_dotenv()
    
    # Initialize the HR Intelligence System
    chat = Chat()
    hr_system = HRIntelligenceSystem(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Add sample documents to the system only if populate_data is True
    if populate_data:
        print("Adding sample documents to the system...")
        populate_job_descriptions(hr_system)
        populate_resumes(hr_system)
        populate_teams(hr_system)
        populate_meeting_notes(hr_system)
        print("Sample data loaded successfully!")
    
    # Interactive query loop
    print("HR Intelligence System Demo")
    print("-----------------------------------")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your HR query: ")
        
        if query.lower() == 'exit':
            break
            
        # Process the query
        try:
            result = chat.ask(query)
            
            print("\nResponse:")
            print("---------")
            print(result.response)
            
            print("\n(Retrieved from", len(result.retrieved_docs), "documents)")
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Check if we should populate data
    populate = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'populate':
        populate = True
        
    run_full_demo(populate_data=populate)