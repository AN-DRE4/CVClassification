from ..base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
import json
import logging
import re
from typing import Dict, Any, List, Optional

# Base system prompt template - will be customized based on configuration
ROLE_SYSTEM_PROMPT_TEMPLATE = """You are an expert CV Analyzer specializing in determining role levels.
For each expertise area identified, determine the appropriate role level:
{role_levels}

Base your assessment on job titles, responsibilities, and duration of experience.
Consider the level of the responsibilities the person has. If some of these responsibilities are at a higher level, then consider leveling up the role.
Provide a confidence score (0-1) and justification for each determination.
Provide an in depth justification for your response. Be clear and concise but also thorough and with a good level of detail.
Format your response as a valid JSON object with "role_levels" as the key containing an array of objects, 
each with "expertise", "level", "confidence", and "justification" fields.
Your entire response/output is going to consist of a single JSON object, and you will NOT wrap it within JSON md markers. This is very important since it will be parsed directly as JSON.{feedback_context}"""

# Default role levels for backward compatibility
DEFAULT_ROLE_LEVELS = [
    {"name": "entry_level", "description": "Junior positions, 0-2 years experience"},
    {"name": "mid_level", "description": "Regular positions, 2-5 years experience"},
    {"name": "senior_level", "description": "Senior positions, 5+ years experience"},
    {"name": "management", "description": "Management positions at any level"}
]

ROLE_USER_PROMPT = """Analyze this CV for role levels:

Work Experience:
{work_experience}

Previously identified expertise areas:
{expertise_results}

Note that the duration in the work experience is in years.

For each expertise area, determine the most appropriate role level with justification.
Your entire response/output is going to consist of a single JSON object, and you will NOT wrap it within JSON md markers.  This is very important since it will be parsed directly as JSON."""

class RoleLevelAgent(BaseAgent):
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, retry_delay=2, custom_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, temperature, max_retries, retry_delay, custom_config)
        self._build_prompt()
        
    def _build_prompt(self):
        """Build the prompt template using current configuration"""
        # Get role levels from config or use defaults
        role_levels = self.custom_config.get("role_levels", DEFAULT_ROLE_LEVELS)

        # Format the role levels as a bullet list
        formatted_role_levels = "\n".join([f"- {level['name']}: {level['description']}" for level in role_levels])
        
        # Get feedback context
        feedback_context = self.get_feedback_context("role_level")
        
        # Create the system prompt with the role levels and feedback
        system_prompt = ROLE_SYSTEM_PROMPT_TEMPLATE.format(
            role_levels=formatted_role_levels,
            feedback_context=feedback_context
        )
        
        # Build the final prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", ROLE_USER_PROMPT)
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
        """Validate role levels result structure"""
        if not isinstance(result, dict):
            return False
        
        if "role_levels" not in result:
            return False
            
        role_levels = result.get("role_levels", [])
        if not isinstance(role_levels, list) or len(role_levels) == 0:
            return False
            
        # Check that at least one role level has required fields
        for role in role_levels:
            if not all(key in role for key in ["expertise", "level", "confidence"]):
                return False
                
        return True
    
    def _get_fallback_result(self, errors):
        """Generate fallback role level result"""
        logging.warning(f"Using fallback result for role level agent after errors: {errors}")
        
        # Check if we received expertise data from the input
        expertise_data = {}
        try:
            # Extract expertise from the most recent error's input if available
            if hasattr(self, 'last_input') and self.last_input and 'expertise_results' in self.last_input:
                expertise_data = self.last_input.get('expertise_results', {})
        except:
            pass
            
        # If we have expertise data, create a generic role level for each expertise
        if expertise_data and "expertise" in expertise_data and isinstance(expertise_data["expertise"], list):
            fallback_roles = []
            for exp in expertise_data["expertise"]:
                if "category" in exp:
                    fallback_roles.append({
                        "expertise": exp["category"],
                        "level": "mid_level",  # Default to mid-level as safest fallback
                        "confidence": 0.5,
                        "justification": "Fallback result due to processing failure"
                    })
            
            if fallback_roles:
                return {"role_levels": fallback_roles}
                
        # Default fallback if we couldn't extract from expertise
        return {
            "role_levels": [
                {
                    "expertise": "general",
                    "level": "mid_level",
                    "confidence": 0.5,
                    "justification": "Failed to determine role levels after multiple attempts"
                }
            ]
        }

    def process(self, cv_data):
        """Process a CV with the agent with retries and input caching"""
        # Store input for potential fallback use
        self.last_input = cv_data
        return super().process(cv_data)

    def _apply_feedback_adjustments(self, result):
        """Apply feedback-based adjustments to role level classifications"""
        if "role_levels" not in result:
            return result
        
        adjusted_role_levels = []
        for role in result["role_levels"]:
            expertise = role["expertise"]
            level = role["level"]
            confidence = role["confidence"]
            
            # Create a key for feedback lookup
            role_key = f"{expertise}_{level}"
            
            # Get feedback summary for this role level
            feedback_summary = self.feedback_manager.get_feedback_summary("role_level", role_key)
            
            # Apply confidence adjustment based on feedback
            confidence_adjustment = feedback_summary.get("confidence_adjustment", 0.0)
            adjusted_confidence = max(0.0, min(1.0, confidence + confidence_adjustment))
            
            # Add feedback information to justification if there's significant feedback
            justification = role.get("justification", "")
            if feedback_summary.get("positive_count", 0) > 0 or feedback_summary.get("negative_count", 0) > 0:
                feedback_info = f" [Confidence adjusted based on {feedback_summary['positive_count']} positive and {feedback_summary['negative_count']} negative user feedback]"
                justification += feedback_info
            
            adjusted_role_levels.append({
                "expertise": expertise,
                "level": level,
                "confidence": adjusted_confidence,
                "justification": justification,
                "original_confidence": confidence,
                "feedback_adjustment": confidence_adjustment
            })
        
        result["role_levels"] = adjusted_role_levels
        return result

def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()