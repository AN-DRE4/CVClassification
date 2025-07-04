from ..base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
import json
import logging
import re
from typing import Dict, Any, List, Optional

# New feedback-oriented system prompt for conversational validation
VALIDATION_FEEDBACK_SYSTEM_PROMPT = """You are a CV Classification Validation Agent specializing in providing detailed feedback on classification results.

CV CONTENT:
{cv_content}

CLASSIFICATION RESULT:
{classification_result}

AGENT TYPE:
{agent_type}

Your role is to:
1. Review classification results from other agents (expertise, role levels, or organizational units)
2. Analyze the quality, accuracy, and appropriateness of each classification
3. Provide constructive feedback indicating what's correct and what needs improvement
4. Decide whether the overall classification is satisfactory or needs further refinement

When reviewing a classification, analyze:
- **Accuracy**: How well do the classifications match the CV content?
- **Evidence**: Is there sufficient evidence in the CV to support each classification?
- **Confidence**: Are the confidence scores appropriate for the evidence provided?
- **Completeness**: Are all relevant classifications included from the available categories?
- **Justifications**: Are the explanations logical and well-reasoned?
- **Category Usage**: Are the classifications using only the available categories listed above?

Provide feedback in this format:
- Identify specific strengths in the classification
- Point out specific issues that need addressing
- Suggest improvements for problematic classifications
- Verify that only available categories are being used
- Indicate overall satisfaction level

Response format: Return a JSON object with:
{{
"validator_satisfied": True | False,
"feedback_summary": "brief overview of your assessment",
"detailed_feedback": "object with feedback for each classification item, in this format (only include the classifications that are present in the current classification_result): 
    {{
    # For expertise classifications from the expertise agent:
    'expertise': {{
        '[expertise_category]': {{
            'accuracy': 'how well the expertise classification matches the CV content',
            'evidence': 'strength of evidence supporting this expertise area',
            'confidence': 'appropriateness of the confidence score given the evidence',
            'completeness': 'whether this expertise area is fully captured',
            'justifications': 'quality and clarity of the justification provided',
            'category_validity': 'whether this expertise category is in the available categories list'
        }},
        '[another_expertise_category]': {{
            'accuracy': 'how well the expertise classification matches the CV content',
            'evidence': 'strength of evidence supporting this expertise area',
            'confidence': 'appropriateness of the confidence score given the evidence',
            'completeness': 'whether this expertise area is fully captured',
            'justifications': 'quality and clarity of the justification provided',
            'category_validity': 'whether this expertise category is in the available categories list'
        }}
        ...
    }},
    
    # For role level classifications from the role levels agent:
    'role_levels': {{
        '[expertise_area]-[role_level]' (Note: the key is a concatenation of the expertise area and its associated role level that the classification result returned): {{
            'accuracy': 'how well the role level matches the candidate\'s experience and responsibilities',
            'evidence': 'strength of evidence supporting this role level assessment',
            'confidence': 'appropriateness of the confidence score for the role level',
            'completeness': 'whether the role level assessment is comprehensive',
            'justifications': 'quality of reasoning for the role level determination',
            'category_validity': 'whether this role level is in the available categories list'
        }},
        '[another_expertise_area]-[another_role_level]' (Note: the key is a concatenation of the expertise area and its associated role level that the classification result returned): {{
            'accuracy': 'how well the role level matches the candidate\'s experience and responsibilities',
            'evidence': 'strength of evidence supporting this role level assessment',
            'confidence': 'appropriateness of the confidence score for the role level',
            'completeness': 'whether the role level assessment is comprehensive',
            'justifications': 'quality of reasoning for the role level determination',
            'category_validity': 'whether this role level is in the available categories list'
        }}
        ...
    }},
    
    # For organizational unit classifications from the organizational units agent:
    'org_unit': {{
        '[org_unit_name]': {{
            'accuracy': 'how well the organizational unit classification fits the candidate\'s background',
            'evidence': 'strength of evidence supporting this organizational unit assignment',
            'confidence': 'appropriateness of the confidence score for the org unit',
            'completeness': 'whether the organizational unit assessment is comprehensive',
            'justifications': 'quality of reasoning for the organizational unit determination',
            'category_validity': 'whether this organizational unit is in the available categories list'
        }},
        '[another_org_unit_name]': {{
            'accuracy': 'how well the organizational unit classification fits the candidate\'s background',
            'evidence': 'strength of evidence supporting this organizational unit assignment',
            'confidence': 'appropriateness of the confidence score for the org unit',
            'completeness': 'whether the organizational unit assessment is comprehensive',
            'justifications': 'quality of reasoning for the organizational unit determination',
            'category_validity': 'whether this organizational unit is in the available categories list'
        }}
        ...
    }}
}}
"strengths": "list of things done well",
"improvements_needed": "list of specific improvements required",
"confidence_assessment": "assessment of whether confidence scores are appropriate",
"category_compliance": "assessment of whether only available categories were used",
"overall_quality": 1-10
}}

Note that the agent that classified the CV may have used different categories than the ones he has access to.
To prevent this, you should check the available categories and make sure that the classification result is using only the available categories.
If there is a category that is not in the available categories, you should not accept the classification result and report it as an error in the category_validity field.
On the other hand, if there is a category in the available ones that makes sense for the classification result, you should report it to the agent.

AVAILABLE CATEGORIES FOR THIS CLASSIFICATION:
{available_categories}

Your entire response must be a valid JSON object without markdown formatting.

Current iteration: {iteration}
Maximum iterations allowed: {max_iterations}

If this is the final iteration (iteration == max_iterations), you should be more lenient and accept reasonable classifications even if not perfect.
"""

class ValidationAgent(BaseAgent):
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, retry_delay=2, custom_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, temperature, max_retries, retry_delay, custom_config)
        
        # Initialize the validation prompt
        self.prompt = ChatPromptTemplate.from_template(VALIDATION_FEEDBACK_SYSTEM_PROMPT)
        
        # Configuration for validation thresholds
        self.satisfaction_threshold = 7.0  # Out of 10 TODO: rever isto, talvez deixar o utilizador definir
        self.minimum_confidence_threshold = 0.3  # 30% minimum confidence
        self.maximum_confidence_threshold = 0.95  # 95% maximum confidence
    
    def provide_feedback(self, cv_data: Dict, agent_type: str, classification_result: Dict, iteration: int, available_categories: Optional[Dict[str, Any]] = None) -> Dict:
        """Provide feedback on a classification result"""
        try:
            # Format available categories for the prompt
            available_categories_text = self._format_available_categories(agent_type, available_categories)

            print("DEBUG: available_categories_text: ", available_categories_text) if agent_type == "role_levels" else None
            
            # Prepare input for the validation LLM
            validation_input = {
                "cv_content": self._extract_cv_content(cv_data),
                "agent_type": agent_type,
                "classification_result": json.dumps(classification_result, indent=2),
                "iteration": iteration,
                "max_iterations": self.max_validation_iterations,
                "available_categories": available_categories_text
            }

            # Format the prompt
            formatted_prompt = self.prompt.format_prompt(**validation_input)
            
            # Get feedback from LLM
            response = self.llm.invoke(formatted_prompt.to_messages())
            feedback = self._parse_response(response.content)
            
            if feedback and not feedback.get("error"):
                # Add metadata
                feedback["agent_type"] = agent_type
                feedback["iteration"] = iteration
                feedback["cv_id"] = cv_data.get("resume_id", "unknown")
                
                # Ensure validator_satisfied is boolean
                if "validator_satisfied" not in feedback:
                    # Fallback: satisfied if overall quality >= threshold
                    overall_quality = feedback.get("overall_quality", 0)
                    feedback["validator_satisfied"] = overall_quality >= self.satisfaction_threshold
                
                return feedback
            else:
                # Fallback feedback if LLM fails
                return self._get_fallback_feedback(classification_result, iteration)
                
        except Exception as e:
            logging.error(f"Error providing validation feedback: {e}")
            return self._get_fallback_feedback(classification_result, iteration)
    
    def _format_available_categories(self, agent_type: str, available_categories: Optional[Dict[str, Any]]) -> str:
        """Format available categories for display in the prompt"""
        if not available_categories:
            return "No category information provided"
        
        if agent_type == "expertise":
            categories = available_categories.get("expertise_categories", [])
            if categories:
                return "Available Expertise Categories:\n" + "\n".join([f"- {cat}" for cat in categories])
        
        elif agent_type == "role_levels":
            role_levels = available_categories.get("role_levels", [])
            expertise_categories = available_categories.get("expertise_categories", [])
            if role_levels:
                formatted_levels = []
                for level in role_levels:
                    name = level.get("name", "unknown")
                    description = level.get("description", "no description")
                    formatted_levels.append(f"- {name}: {description}")
                categories = [f"{cat}-{level}" for cat in expertise_categories for level in role_levels]
                return "Available Role Levels Combinations:\n" + "\n".join(categories)
        
        elif agent_type == "org_unit":
            org_units = available_categories.get("org_units", [])
            if org_units:
                formatted_units = []
                for unit in org_units:
                    name = unit.get("name", "unknown")
                    description = unit.get("description", "no description")
                    formatted_units.append(f"- {name}: {description}")
                return "Available Organizational Units:\n" + "\n".join(formatted_units)
        
        return f"No categories defined for agent type: {agent_type}"
    
    def _extract_cv_content(self, cv_data: Dict) -> str:
        """Extract relevant CV content for validation"""
        content_parts = []
        
        # Add work experience
        work_exp = cv_data.get("work_experience", "")
        if work_exp:
            content_parts.append(f"Work Experience:\n{work_exp}")
        
        # Add skills
        skills = cv_data.get("skills", "")
        if skills:
            content_parts.append(f"Skills:\n{skills}")
        
        # Add education
        education = cv_data.get("education", "")
        if education:
            content_parts.append(f"Education:\n{education}")
        
        # Add resume text if available
        resume_text = cv_data.get("resume_text", "")
        if resume_text and not any(content_parts):  # Only if other sections are empty
            content_parts.append(f"Resume Text:\n{resume_text[:2000]}...")  # Limit length
        
        return "\n\n".join(content_parts) if content_parts else "No CV content available"
    
    def _get_fallback_feedback(self, classification_result: Dict, iteration: int) -> Dict:
        """Provide fallback feedback when LLM fails"""
        # Simple heuristic: check if there are any classifications with very low confidence
        has_low_confidence = False
        
        # Check for low confidence items
        for key in ["expertise", "role_levels", "org_units"]:
            if key in classification_result:
                items = classification_result[key]
                if isinstance(items, list):
                    for item in items:
                        if item.get("confidence", 1.0) < 0.4:
                            has_low_confidence = True
                            break
                elif isinstance(items, dict) and key in items:
                    for item in items[key]:
                        if item.get("confidence", 1.0) < 0.4:
                            has_low_confidence = True
                            break
        
        # Be more lenient on later iterations
        satisfied = not has_low_confidence or iteration >= self.max_validation_iterations
        
        return {
            "validator_satisfied": satisfied,
            "feedback_summary": "Automatic validation due to system error",
            "detailed_feedback": {},
            "strengths": ["Classification completed without errors"],
            "improvements_needed": ["Low confidence items detected"] if has_low_confidence else [],
            "confidence_assessment": "Unable to assess due to system error",
            "overall_quality": 7 if satisfied else 5,
            "agent_type": "validation",
            "iteration": iteration,
            "fallback": True
        }
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse the validation response from the LLM"""
        try:
            # Clean the response text
            cleaned_text = response_text.strip()
            
            # Remove markdown formatting if present
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            
            # Parse JSON
            result = json.loads(cleaned_text)
            
            # Validate required fields
            required_fields = ["validator_satisfied", "feedback_summary"]
            for field in required_fields:
                if field not in result:
                    logging.warning(f"Missing required field in validation response: {field}")
                    if field == "validator_satisfied":
                        result[field] = False
                    else:
                        result[field] = "No information provided"
            
            return result
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse validation JSON response: {e}")
            logging.error(f"Response text: {response_text}")
            return {"error": f"JSON parsing failed: {str(e)}"}
        except Exception as e:
            logging.error(f"Error parsing validation response: {e}")
            return {"error": f"Parsing error: {str(e)}"}
    
    def _validate_result(self, result: Dict) -> bool:
        """Validate the parsed result"""
        if not isinstance(result, dict):
            return False
        
        # Check for error
        if result.get("error"):
            return False
        
        # Check for required fields
        required_fields = ["validator_satisfied", "feedback_summary"]
        return all(field in result for field in required_fields)
    
    def _get_fallback_result(self, errors: List[str]) -> Dict:
        """Provide fallback result when all retries fail"""
        return {
            "error": True,
            "validator_satisfied": False,
            "feedback_summary": "Validation failed due to system errors",
            "detailed_feedback": {},
            "strengths": [],
            "improvements_needed": ["Validation system unavailable"],
            "confidence_assessment": "Unable to assess",
            "overall_quality": 1,
            "details": errors
        }

    # Legacy methods for backward compatibility (though we're moving away from this approach)
    def validate_classification(self, cv_data: Dict, agent_type: str, classification_result: Dict) -> Dict:
        """Legacy method - for backward compatibility only"""
        logging.warning("validate_classification is deprecated. Use provide_feedback instead.")
        
        # Provide feedback using new method
        feedback = self.provide_feedback(cv_data, agent_type, classification_result, 1)
        
        # Convert feedback to old format for compatibility
        if feedback.get("validator_satisfied", False):
            # If satisfied, return original classification
            return classification_result
        else:
            # If not satisfied, add validation info
            result = classification_result.copy()
            result["validation_applied"] = True
            result["validation_feedback"] = feedback
            return result

def clean_json_string(json_string):
    """Clean JSON string by removing markdown formatting"""
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip() 