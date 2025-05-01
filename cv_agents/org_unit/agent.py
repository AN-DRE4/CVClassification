from ..base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
import json

ORG_UNIT_SYSTEM_PROMPT = """You are an expert CV Analyzer specializing in determining optimal organizational units.
Based on the candidate's expertise areas and role levels, determine the most appropriate organizational unit:
- engineering: Software development, DevOps, infrastructure
- data: Data engineering, data science, analytics
- marketing_sales: Marketing, sales, communications
- finance_accounting: Finance, accounting, auditing
- operations: Project management, operations, logistics
- customer_service: Customer support, account management
- hr: Human resources, recruitment, training

Provide a confidence score (0-1) and justification for your determination.
Provide an in depth justification for your response. Be clear and concise but also thorough and with a good level of detail.
Format your response as a valid JSON object. Only return the JSON object, nothing else. This is very important since it will be parsed directly as JSON."""

ORG_UNIT_USER_PROMPT = """Analyze this CV for organizational fit:

Work Experience:
{work_experience}

Skills:
{skills}

Expertise areas:
{expertise_results}

Role levels:
{role_results}

Determine the most appropriate organizational unit(s) with justification.
Provide an in depth justification for your response. Be clear and concise but also thorough and with a good level of detail.
Return the organizational units in JSON format. Only return the JSON object, nothing else. This is very important since it will be parsed directly as JSON."""

class OrgUnitAgent(BaseAgent):
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1):
        super().__init__(model_name, temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ORG_UNIT_SYSTEM_PROMPT),
            ("human", ORG_UNIT_USER_PROMPT)
        ])
    
    def _parse_response(self, response_text):
        """Parse the LLM JSON response"""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"org_units": [{"unit": "unknown", "confidence": 0, "justification": "Failed to parse response"}]}
