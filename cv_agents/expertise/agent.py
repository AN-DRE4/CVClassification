from ..base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
import json
import logging
import re
from typing import Dict, Any, List, Optional

# Base system prompt template - will be customized based on configuration
EXPERTISE_SYSTEM_PROMPT_TEMPLATE = """You are an expert CV Analyzer specializing in identifying areas of expertise.
Analyze the provided CV information and identify the candidate's areas of expertise from categories such as:
{expertise_categories}

If a candidate has experience in multiple areas, you should identify all of them.
Only use the categories listed above and do not make up any categories.
For each identified expertise area, provide a confidence score (0-1) and justification.

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
Format the response as a valid JSON object with "expertise" as the key containing an array of objects, 
each with "category", "confidence", and "justification" fields.
Your entire response/output is going to consist of a single JSON object, and you will NOT wrap it within JSON md markers. This is very important since it will be parsed directly as JSON.
{feedback_context}"""

# Default expertise categories for backward compatibility
DEFAULT_EXPERTISE_CATEGORIES = [
    "software_development",
    "data_engineering",
    "data_science",
    "devops",
    "cybersecurity",
    "marketing",
    "finance",
    "management"
]

EXPERTISE_USER_PROMPT = """Analyze this CV:

Work Experience:
{work_experience}

Skills:
{skills}

Education:
{education}

Note that the duration in the work experience is in years.

{validation_feedback_section}

Your entire response/output is going to consist of a single JSON object, and you will NOT wrap it within JSON md markers. This is very important since it will be parsed directly as JSON.
"""

class ExpertiseAgent(BaseAgent):
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, retry_delay=2, custom_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, temperature, max_retries, retry_delay, custom_config)
        self._build_prompt()
    
    def _build_prompt(self):
        """Build the prompt template using current configuration"""
        # Get expertise categories from config or use defaults
        expertise_categories = self.custom_config.get("expertise_categories", DEFAULT_EXPERTISE_CATEGORIES)
        
        # Format the expertise categories as a bullet list
        formatted_categories = "\n".join([f"- {category}" for category in expertise_categories])
        
        # Get feedback context
        feedback_context = self.get_feedback_context("expertise")
        
        # Create the system prompt with the categories and feedback
        system_prompt = EXPERTISE_SYSTEM_PROMPT_TEMPLATE.format(
            expertise_categories=formatted_categories,
            feedback_context=feedback_context
        )
        
        # Build the final prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", EXPERTISE_USER_PROMPT)
        ])
    
    def _on_config_updated(self):
        """Rebuild prompt when configuration changes"""
        self._build_prompt()
    
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
"""
        cv_data["validation_feedback_section"] = validation_feedback_section
        
        return super().process(cv_data)
    
    def _apply_improved_confidence_scoring(self, result): # TODO: eliminar isto
        """Apply improved confidence scoring with better granularity"""
        if "expertise" not in result:
            return result
        
        improved_expertise = []
        for exp in result["expertise"]:
            category = exp["category"]
            confidence = exp["confidence"]
            justification = exp.get("justification", "")
            
            # Analyze justification and adjust confidence based on evidence strength
            confidence_tier = self._get_confidence_tier(confidence)
            
            # Look for evidence strength indicators in justification
            evidence_strength = self._assess_evidence_strength(justification)
            
            # Adjust confidence based on evidence assessment
            adjusted_confidence = self._adjust_confidence_scoring(confidence, evidence_strength)
            
            # Add evidence assessment to justification
            if evidence_strength != "medium":
                justification += f" [Evidence assessment: {evidence_strength}, confidence tier: {confidence_tier}]"
            
            improved_expertise.append({
                "category": category,
                "confidence": adjusted_confidence,
                "justification": justification,
                "original_confidence": confidence,
                "evidence_strength": evidence_strength,
                "confidence_tier": confidence_tier
            })
        
        result["expertise"] = improved_expertise
        return result
    
    def _assess_evidence_strength(self, justification: str) -> str: # TODO: eliminar isto
        """Assess evidence strength based on justification content"""
        justification_lower = justification.lower()
        
        # Strong evidence indicators
        strong_indicators = ["years of experience", "extensive", "deep expertise", "proven track record", 
                           "multiple projects", "senior", "lead", "manager", "architect"]
        
        # Weak evidence indicators  
        weak_indicators = ["minimal", "limited", "unclear", "indirect", "partial", "brief", 
                          "mentioned", "some exposure", "basic"]
        
        # Very strong evidence indicators
        very_strong_indicators = ["decade", "10+ years", "expert level", "specialized", "advanced", 
                                "comprehensive", "extensive background"]
        
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
        """Apply feedback-based adjustments to expertise classifications using targeted feedback"""
        if "expertise" not in result:
            return result
        
        adjusted_expertise = []
        for exp in result["expertise"]:
            category = exp["category"]
            confidence = exp["confidence"]
            
            # Get targeted feedback summary for this specific category
            feedback_summary = self.feedback_manager.get_feedback_summary("expertise", category)
            
            # Apply confidence adjustment based on targeted feedback
            confidence_adjustment = feedback_summary.get("confidence_adjustment", 0.0)
            adjusted_confidence = max(0.0, min(1.0, confidence + confidence_adjustment))
            
            # Create more detailed justification with feedback context
            justification = exp.get("justification", "")
            
            # Add feedback information if there's meaningful feedback
            feedback_strength = feedback_summary.get("feedback_strength", 0)
            if feedback_strength > 0:
                positive_count = feedback_summary.get("positive_count", 0)
                negative_count = feedback_summary.get("negative_count", 0)
                
                if confidence_adjustment != 0:
                    if confidence_adjustment > 0:
                        feedback_info = f" [Confidence boosted by +{confidence_adjustment:.3f} based on {positive_count} positive user feedback for '{category}']"
                    else:
                        feedback_info = f" [Confidence reduced by {confidence_adjustment:.3f} based on {negative_count} negative user feedback for '{category}']"
                else:
                    feedback_info = f" [Based on {positive_count} positive and {negative_count} negative user feedback for '{category}']"
                
                justification += feedback_info
            
            adjusted_expertise.append({
                "category": category,
                "confidence": adjusted_confidence,
                "justification": justification,
                "original_confidence": confidence,
                "feedback_adjustment": confidence_adjustment
            })
        
        result["expertise"] = adjusted_expertise
        return result

def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()
