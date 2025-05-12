from ..base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
import json
import logging
import re
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
Format your response as a valid JSON object with "org_units" as the key containing an array of objects, 
each with "unit", "confidence", and "justification" fields.
Your entire response/output is going to consist of a single JSON object, and you will NOT wrap it within JSON md markers"""

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
Your entire response/output is going to consist of a single JSON object, and you will NOT wrap it within JSON md markers. This is very important since it will be parsed directly as JSON."""

class OrgUnitAgent(BaseAgent):
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, retry_delay=2):
        super().__init__(model_name, temperature, max_retries, retry_delay)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ORG_UNIT_SYSTEM_PROMPT),
            ("human", ORG_UNIT_USER_PROMPT)
        ])
    
    def _parse_response(self, response_text):
        """Parse the LLM JSON response"""
        try:
            response_text = clean_json_string(response_text)
            return json.loads(response_text)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON response: {response_text}")
            raise ValueError("Response is not valid JSON")
    
    def _validate_result(self, result):
        """Validate org unit result structure"""
        if not isinstance(result, dict):
            return False
        
        if "org_units" not in result:
            return False
            
        org_units = result.get("org_units", [])
        if not isinstance(org_units, list) or len(org_units) == 0:
            return False
            
        # Check that at least one org unit has required fields
        for unit in org_units:
            if not all(key in unit for key in ["unit", "confidence"]):
                return False
                
        return True
    
    def process(self, cv_data):
        """Process a CV with the agent with retries and input caching"""
        # Store input for potential fallback use
        self.last_input = cv_data
        return super().process(cv_data)
    
    def _get_fallback_result(self, errors):
        """Generate fallback org unit result"""
        logging.warning(f"Using fallback result for org unit agent after errors: {errors}")
        
        # Check if we received expertise data from the input
        expertise_data = {}
        try:
            # Extract expertise from the stored input if available
            if hasattr(self, 'last_input') and self.last_input and 'expertise_results' in self.last_input:
                expertise_data = self.last_input.get('expertise_results', {})
        except:
            pass
            
        # Try to infer org unit from expertise areas
        org_unit = "operations"  # Default fallback
        
        # Map expertise categories to org units
        expertise_to_org_unit = {
            "software_development": "engineering",
            "devops": "engineering",
            "data_engineering": "data",
            "data_science": "data",
            "cybersecurity": "engineering",
            "marketing": "marketing_sales",
            "finance": "finance_accounting",
            "management": "operations"
        }
        
        # If we have expertise data, determine the most likely org unit
        if expertise_data and "expertise" in expertise_data and isinstance(expertise_data["expertise"], list):
            # Find the highest confidence expertise
            max_confidence = 0
            top_expertise = None
            
            for exp in expertise_data["expertise"]:
                if "category" in exp and "confidence" in exp:
                    if float(exp["confidence"]) > max_confidence:
                        max_confidence = float(exp["confidence"])
                        top_expertise = exp["category"]
            
            # Map to org unit if available
            if top_expertise in expertise_to_org_unit:
                org_unit = expertise_to_org_unit[top_expertise]
                
        # Return fallback result
        return {
            "org_units": [
                {
                    "unit": org_unit,
                    "confidence": 0.5,
                    "justification": "Fallback organizational unit based on available expertise data"
                }
            ]
        }

def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()