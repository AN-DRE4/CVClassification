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

Only use the role levels provided above. Do not make up your own role levels.
DO NOT IN ANY WAY alter or add to the expertise areas that are provided.
DO NOT IN ANY WAY create role levels for expertise areas that are not provided in the expertise results.

Base your assessment on job titles, responsibilities, and duration of experience.
Consider the level of the responsibilities the person has. If some of these responsibilities are at a higher level, then consider leveling up the role.
Provide a confidence score (0-1) and justification for each determination.

Before making any classification:
1. Carefully analyze the candidate's specific experience and skills
2. Consider the depth and breadth of experience in each area
3. Use improved confidence scoring:
   - Very High confidence (0.90-0.95): Extensive, clear evidence with multiple years of experience
   - High confidence (0.80-0.89): Strong evidence with solid experience and clear skills match
   - Medium High confidence (0.70-0.79): Good evidence with some experience and relevant skills
   - Medium confidence (0.50-0.69): Moderate evidence or indirect indicators
   - Low confidence (0.30-0.49): Minimal evidence or weak indicators
   - Very Low confidence (0.10-0.29): Very little evidence or unclear match

If you receive validation feedback, carefully review the feedback points and adjust your classification accordingly. Pay attention to:
- Specific strengths and weaknesses mentioned
- Evidence gaps identified by the validator
- Confidence level appropriateness feedback

Always explain your reasoning thoroughly and be conservative with confidence scores.
Provide an in depth justification for your response. Be clear and concise but also thorough and with a good level of detail.
Format your response as a valid JSON object with "role_levels" as the key containing an array of objects, 
each with "expertise", "level", "confidence", and "justification" fields.
Your entire response/output is going to consist of a single JSON object, and you will NOT wrap it within JSON md markers. This is very important since it will be parsed directly as JSON.
{feedback_context}"""

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

{validation_feedback_section}

For each expertise area, determine the most appropriate role level with justification.
Only evaluate the role level for the expertise areas that are provided in the expertise results and nothing else.
Your entire response/output is going to consist of a single JSON object, and you will NOT wrap it within JSON md markers.  This is very important since it will be parsed directly as JSON."""

class RoleLevelAgent(BaseAgent):
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, retry_delay=2, custom_config: Optional[Dict[str, Any]] = None, max_validation_iterations: int = 3):
        super().__init__(model_name, temperature, max_retries, retry_delay, custom_config, max_validation_iterations)
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
        
        # Add validation feedback section if present
        validation_feedback_section = ""
        if cv_data.get("validation_feedback"):
            feedback = cv_data["validation_feedback"]
            validation_feedback_section = f"""
VALIDATION FEEDBACK (Iteration {cv_data.get('iteration', 1)}):

{feedback.get('feedback_summary', 'No summary provided')}

DETAILED FEEDBACK:
{feedback.get('detailed_feedback', {})}

STRENGTHS IDENTIFIED:
{', '.join(feedback.get('strengths', []))}

IMPROVEMENTS NEEDED:
{', '.join(feedback.get('improvements_needed', []))}

CONFIDENCE ASSESSMENT:
{feedback.get('confidence_assessment', 'No assessment')}

Please address the feedback points above when revising your classification.
Improve your classification based on the strengths and improvements needed.
Think about the detailed feedback and make your classification more accurate with these points in mind.
"""
        cv_data["validation_feedback_section"] = validation_feedback_section
        
        return super().process(cv_data)

    def _apply_improved_confidence_scoring(self, result): # TODO: rever isto, nao vejo a necessidade de ter isto
        """Apply improved confidence scoring with better granularity"""
        if "role_levels" not in result:
            return result
        
        improved_role_levels = []
        for role in result["role_levels"]:
            expertise = role["expertise"]
            level = role["level"]
            confidence = role["confidence"]
            justification = role.get("justification", "")
            
            # Analyze justification and adjust confidence based on evidence strength
            confidence_tier = self._get_confidence_tier(confidence)
            
            # Look for evidence strength indicators in justification
            evidence_strength = self._assess_evidence_strength(justification)
            
            # Adjust confidence based on evidence assessment
            adjusted_confidence = self._adjust_confidence_scoring(confidence, evidence_strength)
            
            # Add evidence assessment to justification
            if evidence_strength != "medium":
                justification += f" [Evidence assessment: {evidence_strength}, confidence tier: {confidence_tier}]"
            
            improved_role_levels.append({
                "expertise": expertise,
                "level": level,
                "confidence": adjusted_confidence,
                "justification": justification,
                "original_confidence": confidence,
                "original_level": level,  # Store original level assignment
                "evidence_strength": evidence_strength,
                "confidence_tier": confidence_tier
            })
        
        result["role_levels"] = improved_role_levels
        return result
    
    def _assess_evidence_strength(self, justification: str) -> str: # TODO: eliminar isto
        """Assess evidence strength based on justification content"""
        justification_lower = justification.lower()
        
        # Strong evidence indicators for role levels
        strong_indicators = ["years of experience", "extensive", "leadership", "management", "senior position",
                           "team lead", "project lead", "responsible for", "managed", "led", "supervised"]
        
        # Weak evidence indicators  
        weak_indicators = ["minimal", "limited", "unclear", "indirect", "partial", "brief", 
                          "mentioned", "some exposure", "basic", "entry level"]
        
        # Very strong evidence indicators
        very_strong_indicators = ["decade", "10+ years", "director", "vp", "cto", "ceo", "executive",
                                "senior management", "extensive leadership", "proven leadership"]
        
        # Count indicators
        strong_count = sum(1 for indicator in strong_indicators if indicator in justification_lower)
        weak_count = sum(1 for indicator in weak_indicators if indicator in justification_lower)  
        very_strong_count = sum(1 for indicator in very_strong_indicators if indicator in justification_lower)
        
        if very_strong_count > 0:
            return "very_strong"
        elif strong_count > weak_count and strong_count > 0:
            return "strong"
        elif weak_count > strong_count and weak_count > 0:
            return "weak"
        elif weak_count > 1:
            return "very_weak"
        else:
            return "medium"

    def _apply_feedback_adjustments(self, result):
        """Apply feedback-based adjustments to role level classifications using targeted feedback"""
        if "role_levels" not in result:
            return result
        
        adjusted_role_levels = []
        for role in result["role_levels"]:
            expertise = role["expertise"]
            level = role["level"]
            confidence = role["confidence"]
            
            # Create a key for targeted feedback lookup
            role_key = f"{expertise}_{level}"
            
            # Get targeted feedback summary for this specific role level
            feedback_summary = self.feedback_manager.get_feedback_summary("role_level", role_key)
            
            # Apply confidence adjustment based on targeted feedback
            confidence_adjustment = feedback_summary.get("confidence_adjustment", 0.0)
            adjusted_confidence = max(0.0, min(1.0, confidence + confidence_adjustment))
            
            # Create more detailed justification with feedback context
            justification = role.get("justification", "")
            
            # Add feedback information if there's meaningful feedback
            feedback_strength = feedback_summary.get("feedback_strength", 0)
            if feedback_strength > 0:
                positive_count = feedback_summary.get("positive_count", 0)
                negative_count = feedback_summary.get("negative_count", 0)
                
                if confidence_adjustment != 0:
                    if confidence_adjustment > 0:
                        feedback_info = f" [Confidence boosted by +{confidence_adjustment:.3f} based on {positive_count} positive user feedback for '{expertise}-{level}']"
                    else:
                        feedback_info = f" [Confidence reduced by {confidence_adjustment:.3f} based on {negative_count} negative user feedback for '{expertise}-{level}']"
                else:
                    feedback_info = f" [Based on {positive_count} positive and {negative_count} negative user feedback for '{expertise}-{level}']"
                
                justification += feedback_info
            
            # Only set original_confidence and original_level if they haven't been set by the base agent's tracking
            item_data = {
                "expertise": expertise,
                "level": level,
                "confidence": adjusted_confidence,
                "justification": justification,
                "feedback_adjustment": confidence_adjustment
            }
            
            # Preserve original_confidence and original_level if they were already set by the validation tracker
            if "original_confidence" not in role:
                item_data["original_confidence"] = confidence
            else:
                item_data["original_confidence"] = role["original_confidence"]
            
            if "original_level" not in role:
                item_data["original_level"] = level
            else:
                item_data["original_level"] = role["original_level"]
            
            adjusted_role_levels.append(item_data)
        
        result["role_levels"] = adjusted_role_levels
        return result

def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()