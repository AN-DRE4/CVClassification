from ..base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
import json

EXPERTISE_SYSTEM_PROMPT = """You are an expert CV Analyzer specializing in identifying areas of expertise.
Analyze the provided CV information and identify the candidate's areas of expertise from these categories:
- software_development
- data_engineering
- data_science
- devops
- cybersecurity
- marketing
- finance
- management

If a candidate has experience in multiple areas, you should identify all of them.
If a candidate has expertise in a field that is not listed above, identify it using the actual category name.
For each identified expertise area, provide a confidence score (0-1) and justification.
Format the response as a valid JSON object with "expertise" as the key containing an array of objects, 
each with "category", "confidence", and "justification" fields."""

EXPERTISE_USER_PROMPT = """Analyze this CV:

Work Experience:
{work_experience}

Skills:
{skills}

Education:
{education}

Return the expertise areas identified in JSON format. Only return the JSON object, nothing else. This is very important since it will be parsed directly as JSON."""

class ExpertiseAgent(BaseAgent):
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1):
        super().__init__(model_name, temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EXPERTISE_SYSTEM_PROMPT),
            ("human", EXPERTISE_USER_PROMPT)
        ])
    
    def _parse_response(self, response_text):
        """Parse the LLM JSON response"""
        try:
            response_json = json.loads(response_text)
            return response_json
        except json.JSONDecodeError:
            # Fallback parsing if response isn't valid JSON
            # Implement regex or other extraction methods here
            return {"expertise": [{"category": "unknown", "confidence": 0, "justification": "Failed to parse response"}]}
