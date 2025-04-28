import json
from openai import OpenAI


class MultiStageMCP:
    def __init__(self, openai_api_key, model="gpt-4"):
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        
        # Base system prompt that will be used across all stages
        self.base_system_prompt = """
You are an HR Intelligence Assistant, specialized in helping HR professionals find information 
about job descriptions, team structures, candidate resumes, and meeting notes.
"""

    def generate_response(self, query, context):
        """Generate a response using multi-stage MCP approach"""
        # Stage 1: Analyze the query to understand what information is needed
        query_analysis = self._stage_analyze_query(query)
        
        # Stage 2: Analyze the retrieved context to extract relevant information
        context_analysis = self._stage_analyze_context(query, query_analysis, context)
        
        # Stage 3: Generate the final response using insights from both previous stages
        final_response = self._stage_generate_response(query, query_analysis, context_analysis)
        
        return final_response

    def _stage_analyze_query(self, query):
        """Stage 1: Analyze the query to understand what information is needed"""
        prompt = self.base_system_prompt + """
In this stage, you will analyze the user's query to understand:
1. What type of information they are seeking (job descriptions, team structures, resumes, meeting notes)
2. Any specific entities mentioned (specific job titles, teams, candidates, skills)
3. The intent behind the query (matching candidates to jobs, finding information about teams, etc.)
4. Additional context that might be needed to fully answer the question

Provide your analysis in a structured format with clear reasoning.
"""
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Please analyze the following HR query: \"{query}\""}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content

    def _stage_analyze_context(self, query, query_analysis, context):
        """Stage 2: Analyze the retrieved context to extract relevant information"""
        prompt = self.base_system_prompt + """
In this stage, you will analyze the retrieved context in relation to the user's query and the query analysis.
Your task is to:

1. Identify which documents in the context are most relevant to the query
2. Extract key information from those documents that helps answer the query
3. Note any missing information that would be needed for a complete answer
4. Identify connections between different documents (e.g., skills in a resume matching requirements in a job description)

Focus on being precise and factual, using only information provided in the context.
"""
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"""
User query: {query}

Query analysis:
{query_analysis}

Retrieved context:
{context}

Please analyze the context in relation to this query and identify the most relevant information.
"""}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        
        return response.choices[0].message.content

    def _stage_generate_response(self, query, query_analysis, context_analysis):
        """Stage 3: Generate the final response using insights from both previous stages"""
        prompt = self.base_system_prompt + """
Now you will generate a final response to the user's query.

Guidelines:
1. Only use information mentioned in the context analysis
2. Format your response in a clear, professional manner suitable for HR specialists
3. If the context doesn't contain enough information, acknowledge this in your response
4. For job-candidate matching, highlight specific qualifications that match requirements
5. Be direct and concise while providing all relevant information
6. Do not make up information that isn't in the context analysis
"""
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"""
User query: {query}

Query analysis:
{query_analysis}

Context analysis:
{context_analysis}

Please provide a helpful response to the HR professional's query.
"""}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content