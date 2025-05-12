from ..base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
import json
import logging
import re

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
Provide an in depth justification for your response. Be clear and concise but also thorough and with a good level of detail.
Format the response as a valid JSON object with "expertise" as the key containing an array of objects, 
each with "category", "confidence", and "justification" fields.
Your entire response/output is going to consist of a single JSON object, and you will NOT wrap it within JSON md markers. This is very important since it will be parsed directly as JSON."""

EXPERTISE_USER_PROMPT = """Analyze this CV:

Work Experience:
{work_experience}

Skills:
{skills}

Education:
{education}

Your entire response/output is going to consist of a single JSON object, and you will NOT wrap it within JSON md markers. This is very important since it will be parsed directly as JSON."""

class ExpertiseAgent(BaseAgent):
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, retry_delay=2):
        super().__init__(model_name, temperature, max_retries, retry_delay)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EXPERTISE_SYSTEM_PROMPT),
            ("human", EXPERTISE_USER_PROMPT)
        ])
    
    def _parse_response(self, response_text):
        """Parse the LLM JSON response"""
        try:
            response_text = clean_json_string(response_text)
            response_json = json.loads(response_text)
            return response_json
        except json.JSONDecodeError:
            # Fallback parsing if response isn't valid JSON
            logging.error(f"Failed to parse JSON response: {response_text}")
            raise ValueError("Response is not valid JSON")
    
    def _validate_result(self, result):
        """Validate expertise result structure"""
        if not isinstance(result, dict):
            return False
        
        if "expertise" not in result:
            return False
            
        expertise_list = result.get("expertise", [])
        if not isinstance(expertise_list, list) or len(expertise_list) == 0:
            return False
            
        # Check that at least one expertise has required fields
        for exp in expertise_list:
            if not all(key in exp for key in ["category", "confidence"]):
                return False
                
        return True
    
    def _get_fallback_result(self, errors):
        """Generate fallback expertise result"""
        logging.warning(f"Using fallback result for expertise agent after errors: {errors}")
        return {
            "expertise": [
                {
                    "category": "unknown", 
                    "confidence": 0.5, 
                    "justification": "Failed to identify expertise after multiple attempts"
                }
            ]
        }

    def process(self, cv_data):
        """Process a CV with the agent with retries and input caching"""
        # Store input for potential fallback use
        self.last_input = cv_data
        return super().process(cv_data)
    
def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()
