
from app.ai_service import AiServicePrompt, GenimiAiService
from app.configs import FAST_GEMINI_MODEL
from app.schemas.base import AiResponseBaseModel
from app.schemas.retriever import Entity, InformationType, RetrieverQueryAnalysis, RetrieverOptimizedQueries


class QueryAnalysisService:
    def __init__(self):
        self.ai_service = GenimiAiService(FAST_GEMINI_MODEL)

    def analyze_query(self, query: str) -> RetrieverQueryAnalysis:
        prompt = f"""
You are an HR Intelligence Assistant analyzing a user query to understand what information is needed from ChromaDB.

Analyze the following HR query to determine:
1. What type of information is being sought. Possible values are: {[x.value for x in InformationType]}. DON'T USE any other values.
2. Any specific entities mentioned. Possible values are: {[x.value for x in Entity]}
3. The intent behind the query (matching candidates to jobs, finding information about teams, etc.)
4. Key terms that would be important for vector similarity search in ChromaDB
5. Any implicit information needs not directly mentioned in the query

IMPORTANT: If the query mentions ANY person's name
- Always include the name in the people_identified list
- Make sure the intent reflects that we're looking for information about this specific person
- Person names are critical signals for ChromaDB retrieval

Respond *only* with a valid JSON object matching the provided schema. Do not add any introductory text, explanations outside the JSON structure, or markdown formatting.
"""
        user_prompt = f"Analyze this HR query for ChromaDB retrieval: \"{query}\""

        prompt = AiServicePrompt(
            system_prompt=prompt,
            user_prompt=user_prompt
        )

        result = self.ai_service.generate_response(
            prompt=prompt, response_model=RetrieverQueryAnalysis)
        return result

    def improve_query(self, query: str) -> str:
        prompt = """
You are an HR Intelligence Assistant that specializes in reformulating user queries to make them more effective for retrieving relevant information from ChromaDB, a vector database that uses semantic similarity.

IMPORTANT CHROMADB OPTIMIZATION GUIDELINES:
1. Create concise, focused queries (4-8 words is often ideal)
2. Use key domain-specific terms that would appear in relevant documents
3. Avoid unnecessary filler words and focus on meaningful keywords
4. Include semantic variations of key terms to improve vector matching
5. For person queries, prioritize the full name and key identifiers

If the query mentions ANY person's name, create a query that:
1. Places the person's name at the beginning for maximum weight
2. Uses the most complete form of the name available, yet DON'T make up any new names
3. Includes 1-2 key HR terms most relevant to the information need (resume, interview, qualifications)

For example:
- "Find information about {PERSON_NAME}" → "{PERSON_NAME} candidate profile resume"
- "Draft an email about job openings for {PERSON_NAME}" → "{PERSON_NAME} skills qualifications job match"

For non-person queries, optimize by:
1. Including domain-specific terminology highly likely to exist in target documents
2. Removing words that add little semantic value (articles, common verbs)
3. Structuring from most to least important concepts (for ChromaDB's embedding behavior)

Your output should be JUST THE ENHANCED QUERY with no explanations or formatting.
"""
        user_prompt = f"Original query: {query}\n\nEnhance this query for ChromaDB retrieval:"

        prompt_obj = AiServicePrompt(
            system_prompt=prompt,
            user_prompt=user_prompt
        )

        class ImprovedQuery(AiResponseBaseModel):
            improved_query: str

        improved_query = self.ai_service.generate_response(
            response_model=ImprovedQuery,
            prompt=prompt_obj,
        )

        return improved_query.improved_query

    def generate_person_queries(self, original_query: str, query_analysis: RetrieverQueryAnalysis) -> RetrieverOptimizedQueries:
        prompt = '''
You are an HR Intelligence Assistant working with ChromaDB for document retrieval.

Generate 3-5 VARIATIONS focusing on the person, optimized for vector similarity search:

1. Keep queries concise (4-8 words is ideal for ChromaDB)
2. Start with the person's name to give it maximum weight in the embedding
3. Include key HR terms that would likely appear in relevant documents
4. Vary the terms to increase retrieval chance across different document phrasing
5. Avoid filler words to maximize signal-to-noise ratio

EXAMPLE VARIATIONS:
- "NAME resume experience background"
- "NAME qualifications skills history"
- "NAME interview feedback assessment"
- "NAME performance evaluation ratings"

Respond *only* with a valid JSON object matching the provided schema. Do not add any introductory text, explanations outside the JSON structure, or markdown formatting.
'''

        user_prompt = f'''
# Input
Original query: {original_query}

Query analysis: {query_analysis.model_dump_json()}
'''

        prompt = AiServicePrompt(
            system_prompt=prompt,
            user_prompt=user_prompt
        )

        result = self.ai_service.generate_response(
            prompt=prompt, response_model=RetrieverOptimizedQueries)
        return result

    def generate_team_queries(self, original_query: str, query_analysis: RetrieverQueryAnalysis) -> RetrieverOptimizedQueries:
        prompt = '''
You are an HR Intelligence Assistant working with ChromaDB for document retrieval.

Generate 3-5 VARIATIONS focusing on the team, optimized for vector similarity search:

1. Keep queries concise (4-8 words is ideal for ChromaDB)
2. Start with the team name to give it maximum weight in the embedding
3. Include key terms likely to appear in relevant team documents
4. Vary terminology to increase retrieval chance across different document phrasing
5. Don't include person's name in the query, if a person is mentioned

EXAMPLE VARIATIONS:
- "TEAM structure organization composition members"
- "TEAM responsibilities projects deliverables"
- "TEAM performance metrics achievements"
- "TEAM skills experience domain"

Respond *only* with a valid JSON object matching the provided schema. Do not add any introductory text, explanations outside the JSON structure, or markdown formatting.
'''

        user_prompt = f'''
# Input
Original query: {original_query}

Query analysis: {query_analysis.model_dump_json()}
'''

        prompt = AiServicePrompt(
            system_prompt=prompt,
            user_prompt=user_prompt
        )

        result = self.ai_service.generate_response(
            prompt=prompt, response_model=RetrieverOptimizedQueries)
        return result

    def generate_job_queries(self, original_query: str, query_analysis: RetrieverQueryAnalysis) -> RetrieverOptimizedQueries:
        prompt = '''
You are an HR Intelligence Assistant working with ChromaDB for document retrieval.

Generate 3-5 VARIATIONS focusing on the job, optimized for vector similarity search:

1. Keep queries concise (4-8 words is ideal for ChromaDB)
2. Start with the specific job title to give it maximum weight
3. Include key terms likely to appear in relevant job documents
4. Vary the terminology to increase retrieval chance across different document phrasing
5. Avoid filler words that dilute the embedding

EXAMPLE VARIATIONS:
- "JOB_TITLE requirements qualifications responsibilities"
- "JOB_TITLE skills experience needed"
- "JOB_TITLE description role details"
- "JOB_TITLE compensation level salary"

Respond *only* with a valid JSON object matching the provided schema. Do not add any introductory text, explanations outside the JSON structure, or markdown formatting.
'''

        user_prompt = f'''
# Input
Original query: {original_query}

Query analysis: {query_analysis.model_dump_json()}
'''

        prompt = AiServicePrompt(
            system_prompt=prompt,
            user_prompt=user_prompt
        )
        
        result = self.ai_service.generate_response(
            prompt=prompt, response_model=RetrieverOptimizedQueries)
        return result

    def generate_interview_queries(self, original_query: str, query_analysis: RetrieverQueryAnalysis) -> RetrieverOptimizedQueries:
        prompt = '''
You are an HR Intelligence Assistant working with ChromaDB for document retrieval.

Generate 3-5 VARIATIONS focusing on the interview, optimized for vector similarity search:

1. Keep queries concise (4-8 words is ideal for ChromaDB)
2. If a person is mentioned, start with their name for maximum weight
3. Include key interview-related terms that would appear in relevant documents
4. Vary terminology to increase retrieval chance across different document phrasing
5. Focus on specific interview aspects (feedback, assessment, ratings) 

EXAMPLE VARIATIONS:
- "NAME interview feedback evaluation assessment"
- "NAME performance ratings strengths weaknesses"
- "NAME technical skills assessment interview"
- "NAME cultural fit team compatibility"

Respond *only* with a valid JSON object matching the provided schema. Do not add any introductory text, explanations outside the JSON structure, or markdown formatting.
'''

        user_prompt = f'''
# Input
Original query: {original_query}

Query analysis: {query_analysis.model_dump_json()}
'''

        prompt = AiServicePrompt(
            system_prompt=prompt,
            user_prompt=user_prompt
        )

        result = self.ai_service.generate_response(
            prompt=prompt, response_model=RetrieverOptimizedQueries)
        return result