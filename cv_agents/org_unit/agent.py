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
Format your response as a valid JSON object with "org_units" as the key containing an array of objects, 
each with "unit", "confidence", and "justification" fields.
Your entire response/output is going to consist of a single JSON object, and you will NOT wrap it within JSON md markers.
{feedback_context}"""

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

Note that the duration in the work experience is in years.

{validation_feedback_section}

Determine the most appropriate organizational unit(s) with justification.
Provide an in depth justification for your response. Be clear and concise but also thorough and with a good level of detail.
Your entire response/output is going to consist of a single JSON object, and you will NOT wrap it within JSON md markers. This is very important since it will be parsed directly as JSON."""

class OrgUnitAgent(BaseAgent):
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, retry_delay=2, custom_config: Optional[Dict[str, Any]] = None, max_validation_iterations: int = 3):
        super().__init__(model_name, temperature, max_retries, retry_delay, custom_config, max_validation_iterations)
        self._build_prompt()
        
    def _build_prompt(self):
        """Build the prompt template using current configuration"""
        # Get org units from config or use defaults
        org_units = self.custom_config.get("org_units", DEFAULT_ORG_UNITS)
        
        # Format the org units as a bullet list
        formatted_org_units = "\n".join([f"- {unit['name']}: {unit['description']}" for unit in org_units])
        
        # Get feedback context
        feedback_context = self.get_feedback_context("org_unit")
        
        # Create the system prompt with the org units and feedback
        system_prompt = ORG_UNIT_SYSTEM_PROMPT_TEMPLATE.format(
            org_units=formatted_org_units,
            feedback_context=feedback_context
        )
        
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

    def _apply_improved_confidence_scoring(self, result): # TODO: rever isto, nao vejo a necessidade de ter isto
        """Apply improved confidence scoring with better granularity"""
        if "org_units" not in result:
            return result
        
        improved_org_units = []
        for unit in result["org_units"]:
            unit_name = unit["unit"]
            confidence = unit["confidence"]
            justification = unit.get("justification", "")
            
            # Analyze justification and adjust confidence based on evidence strength
            confidence_tier = self._get_confidence_tier(confidence)
            
            # Look for evidence strength indicators in justification
            evidence_strength = self._assess_evidence_strength(justification)
            
            # Adjust confidence based on evidence assessment
            adjusted_confidence = self._adjust_confidence_scoring(confidence, evidence_strength)
            
            # Add evidence assessment to justification
            if evidence_strength != "medium":
                justification += f" [Evidence assessment: {evidence_strength}, confidence tier: {confidence_tier}]"
            
            improved_org_units.append({
                "unit": unit_name,
                "confidence": adjusted_confidence,
                "justification": justification,
                "original_confidence": confidence,
                "evidence_strength": evidence_strength,
                "confidence_tier": confidence_tier
            })
        
        result["org_units"] = improved_org_units
        return result
    
    def _assess_evidence_strength(self, justification: str) -> str: # TODO: eliminar isto
        """Assess evidence strength based on justification content"""
        justification_lower = justification.lower()
        
        # Strong evidence indicators for organizational units
        strong_indicators = ["years of experience", "extensive", "department", "team", "division",
                           "worked in", "experience in", "background in", "specialized in", "focused on"]
        
        # Weak evidence indicators  
        weak_indicators = ["minimal", "limited", "unclear", "indirect", "partial", "brief", 
                          "mentioned", "some exposure", "basic", "transferable"]
        
        # Very strong evidence indicators
        very_strong_indicators = ["decade", "10+ years", "department head", "team lead", "manager",
                                "director", "specialized team", "dedicated role", "core function"]
        
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
        """Apply feedback-based adjustments to org unit classifications using targeted feedback"""
        if "org_units" not in result:
            return result
        
        adjusted_org_units = []
        for unit in result["org_units"]:
            unit_name = unit["unit"]
            confidence = unit["confidence"]
            
            # Get targeted feedback summary for this specific org unit
            feedback_summary = self.feedback_manager.get_feedback_summary("org_unit", unit_name)
            
            # Apply confidence adjustment based on targeted feedback
            confidence_adjustment = feedback_summary.get("confidence_adjustment", 0.0)
            adjusted_confidence = max(0.0, min(1.0, confidence + confidence_adjustment))
            
            # Create more detailed justification with feedback context
            justification = unit.get("justification", "")
            
            # Add feedback information if there's meaningful feedback
            feedback_strength = feedback_summary.get("feedback_strength", 0)
            if feedback_strength > 0:
                positive_count = feedback_summary.get("positive_count", 0)
                negative_count = feedback_summary.get("negative_count", 0)
                
                if confidence_adjustment != 0:
                    if confidence_adjustment > 0:
                        feedback_info = f" [Confidence boosted by +{confidence_adjustment:.3f} based on {positive_count} positive user feedback for '{unit_name}']"
                    else:
                        feedback_info = f" [Confidence reduced by {confidence_adjustment:.3f} based on {negative_count} negative user feedback for '{unit_name}']"
                else:
                    feedback_info = f" [Based on {positive_count} positive and {negative_count} negative user feedback for '{unit_name}']"
                
                justification += feedback_info
            
            # Only set original_confidence if it hasn't been set by the base agent's tracking
            item_data = {
                "unit": unit_name,
                "confidence": adjusted_confidence,
                "justification": justification,
                "feedback_adjustment": confidence_adjustment
            }
            
            # Preserve original_confidence if it was already set by the validation tracker
            if "original_confidence" not in unit:
                item_data["original_confidence"] = confidence
            else:
                item_data["original_confidence"] = unit["original_confidence"]
            
            adjusted_org_units.append(item_data)
        
        result["org_units"] = adjusted_org_units
        return result

def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()