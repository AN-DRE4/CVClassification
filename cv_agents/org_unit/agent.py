from ..base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
import json
import logging
import re
from typing import Dict, Any, List, Optional

# Base system prompt template - will be customized based on configuration
ORG_UNIT_SYSTEM_PROMPT_TEMPLATE = """You are an expert CV Analyzer specializing in determining optimal organizational units.
Based on the candidate's expertise areas and role levels, determine the most appropriate organizational unit:
{org_units}

Provide a confidence score (0-1) and justification for your determination.
Provide an in depth justification for your response. Be clear and concise but also thorough and with a good level of detail.
Format your response as a valid JSON object with "org_units" as the key containing an array of objects, 
each with "unit", "confidence", and "justification" fields.
Your entire response/output is going to consist of a single JSON object, and you will NOT wrap it within JSON md markers"""

# Default org units for backward compatibility
DEFAULT_ORG_UNITS = [
    {"name": "engineering", "description": "Software development, DevOps, infrastructure"},
    {"name": "data", "description": "Data engineering, data science, analytics"},
    {"name": "marketing_sales", "description": "Marketing, sales, communications"},
    {"name": "finance_accounting", "description": "Finance, accounting, auditing"},
    {"name": "operations", "description": "Project management, operations, logistics"},
    {"name": "customer_service", "description": "Customer support, account management"},
    {"name": "hr", "description": "Human resources, recruitment, training"}
]

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
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, retry_delay=2, custom_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, temperature, max_retries, retry_delay, custom_config)
        self._build_prompt()
        
    def _build_prompt(self):
        """Build the prompt template using current configuration"""
        # Get org units from config or use defaults
        org_units = self.custom_config.get("org_units", DEFAULT_ORG_UNITS)
        
        # Format the org units as a bullet list
        formatted_org_units = "\n".join([f"- {unit['name']}: {unit['description']}" for unit in org_units])
        
        # Create the system prompt with the org units
        system_prompt = ORG_UNIT_SYSTEM_PROMPT_TEMPLATE.format(org_units=formatted_org_units)
        
        # Build the final prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", ORG_UNIT_USER_PROMPT)
        ])
    
    def _on_config_updated(self):
        """Rebuild prompt when configuration changes"""
        self._build_prompt()
    
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
        expertise_to_org_unit = self._get_expertise_to_org_unit_mapping()
        
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
    
    def _get_expertise_to_org_unit_mapping(self):
        """Get a mapping of expertise categories to org units based on current configuration"""
        # Default mapping for backward compatibility
        default_mapping = {
            "software_development": "engineering",
            "devops": "engineering",
            "data_engineering": "data",
            "data_science": "data",
            "cybersecurity": "engineering",
            "marketing": "marketing_sales",
            "finance": "finance_accounting",
            "management": "operations"
        }
        
        # If we have custom org units, try to build a mapping
        if "org_units" in self.custom_config:
            # Initialize an empty mapping
            mapping = {}
            
            # Get the org units
            org_units = self.custom_config["org_units"]
            
            # Try to extract mapping from descriptions if available
            for unit in org_units:
                if "name" in unit and "description" in unit:
                    # Extract keywords from description
                    description = unit["description"].lower()
                    name = unit["name"]
                    
                    # Basic keyword matching - this could be enhanced
                    if "software" in description or "development" in description:
                        mapping["software_development"] = name
                    if "data" in description and "engineering" in description:
                        mapping["data_engineering"] = name
                    if "data" in description and "science" in description:
                        mapping["data_science"] = name
                    if "devops" in description:
                        mapping["devops"] = name
                    if "security" in description or "cyber" in description:
                        mapping["cybersecurity"] = name
                    if "marketing" in description:
                        mapping["marketing"] = name
                    if "finance" in description:
                        mapping["finance"] = name
                    if "management" in description:
                        mapping["management"] = name
            
            # Return the custom mapping with defaults as fallback
            return {**default_mapping, **mapping}
        
        return default_mapping

def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()