from ..base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
import json

ROLE_SYSTEM_PROMPT = """You are an expert CV Analyzer specializing in determining role levels.
For each expertise area identified, determine the appropriate role level:
- entry_level: Junior positions, 0-2 years experience
- mid_level: Regular positions, 2-5 years experience
- senior_level: Senior positions, 5+ years experience
- management: Management positions at any level

Base your assessment on job titles, responsibilities, and duration of experience.
Consider the level of the responsibilities the person has. If some of these responsibilities are at a higher level, then consider leveling up the role.
Provide a confidence score (0-1) and justification for each determination.
Provide an in depth justification for your response. Be clear and concise but also thorough and with a good level of detail.
Format your response as a valid JSON object."""

ROLE_USER_PROMPT = """Analyze this CV for role levels:

Work Experience:
{work_experience}

Previously identified expertise areas:
{expertise_results}

For each expertise area, determine the most appropriate role level with justification.
Return the role levels in JSON format. Only return the JSON object, nothing else. This is very important since it will be parsed directly as JSON."""

class RoleLevelAgent(BaseAgent):
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1):
        super().__init__(model_name, temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ROLE_SYSTEM_PROMPT),
            ("human", ROLE_USER_PROMPT)
        ])
    
    def _parse_response(self, response_text):
        """Parse the LLM JSON response"""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"role_levels": [{"expertise": "unknown", "level": "unknown", "confidence": 0, "justification": "Failed to parse response"}]}
