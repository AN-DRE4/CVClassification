from ..base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
import json
import logging
import re
from typing import Dict, Any, List, Optional

VALIDATION_SYSTEM_PROMPT = """You are a CV Classification Validation Agent specializing in reviewing and correcting low-confidence classifications.

Your role is to evaluate classifications that have confidence scores below 80% and determine if they are correct or need to be corrected.

When reviewing a classification, you should:
1. Carefully analyze the CV content and the proposed classification
2. Check if the justification provided makes logical sense given the CV content
3. Determine if the confidence score is appropriate for the evidence presented
4. Either VALIDATE the classification (if correct) or CORRECT it (if incorrect)

For corrections, you should:
- Provide a new classification with proper confidence and justification
- Do not alter the classification category, expertise, level or unit, only the confidence and justification
- If the category, expertise, level or unit is not correct, simply provide an even lower confidence score and a justification for why it is not correct
- The confidence score should be between 0 and 1
- Explain why the original classification was incorrect
- Be conservative with confidence scores - only use high confidence when evidence is clear

For validations, you should:
- Confirm why the classification is correct despite low confidence
- Optionally adjust the confidence score if it was too conservative
- Provide additional justification if helpful

Response format: Return a JSON object with:
- "action": "validate" or "correct"
- "validated_classification": the final classification (corrected or validated)
- "validation_reason": explanation of your decision
- "original_confidence": the original confidence score
- "final_confidence": the final confidence score
- "confidence_change": the change in confidence (positive or negative)

Your entire response must be a valid JSON object without markdown formatting.
"""

VALIDATION_USER_PROMPT = """Please review this CV classification:

CV Content:
Work Experience: {work_experience}
Skills: {skills}
Education: {education}

Original Classification:
Agent Type: {agent_type}
Classification: {classification}
Confidence: {confidence}
Justification: {justification}

Please validate or correct this classification based on the CV content provided.
"""

class ValidationAgent(BaseAgent):
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, retry_delay=2, custom_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, temperature, max_retries, retry_delay, custom_config)
        self._build_prompt()
    
    def _build_prompt(self):
        """Build the prompt template for validation"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", VALIDATION_SYSTEM_PROMPT),
            ("human", VALIDATION_USER_PROMPT)
        ])
    
    def validate_classification(self, cv_data: Dict, agent_type: str, classification_result: Dict) -> Dict:
        """Validate a single classification result"""
        # Check if validation is needed (confidence < 0.8)
        if not self._needs_validation(classification_result, agent_type):
            return classification_result
        
        # Extract low-confidence items
        low_confidence_items = self._extract_low_confidence_items(classification_result, agent_type)
        
        if not low_confidence_items:
            return classification_result
        
        # Validate each low-confidence item
        validated_result = classification_result.copy()
        validation_feedback = []
        
        for item in low_confidence_items:
            validation_input = {
                "work_experience": cv_data.get("work_experience", ""),
                "skills": cv_data.get("skills", ""),
                "education": cv_data.get("education", ""),
                "agent_type": agent_type,
                "classification": json.dumps(item["classification"], indent=2),
                "confidence": item["confidence"],
                "justification": item["justification"]
            }
            
            try:
                # Get validation from LLM
                validation_response = self.process(validation_input)
                
                if validation_response and not validation_response.get("error"):
                    # Apply the validation result
                    validated_result = self._apply_validation_result(
                        validated_result, item, validation_response, agent_type
                    )
                    
                    # Create feedback for the feedback manager
                    feedback = self._create_validation_feedback(
                        cv_data.get("resume_id", ""), item, validation_response, agent_type
                    )
                    validation_feedback.append(feedback)
                    
            except Exception as e:
                logging.error(f"Error validating classification: {e}")
                continue
        
        # Add validation metadata
        validated_result["validation_applied"] = True
        validated_result["validation_feedback"] = validation_feedback
        
        return validated_result
    
    def _needs_validation(self, classification_result: Dict, agent_type: str) -> bool:
        """Check if any classification needs validation (confidence < 0.8)"""
        if agent_type == "expertise":
            expertise_list = classification_result.get("expertise", [])
            return any(exp.get("confidence", 1.0) < 0.8 for exp in expertise_list)
        
        elif agent_type == "role_levels":
            role_list = classification_result.get("role_levels", [])
            return any(role.get("confidence", 1.0) < 0.8 for role in role_list)
        
        elif agent_type == "org_unit":
            org_list = classification_result.get("org_units", [])
            return any(org.get("confidence", 1.0) < 0.8 for org in org_list)
        
        return False
    
    def _extract_low_confidence_items(self, classification_result: Dict, agent_type: str) -> List[Dict]:
        """Extract items with confidence < 0.8"""
        low_confidence_items = []
        
        if agent_type == "expertise":
            expertise_list = classification_result.get("expertise", [])
            for i, exp in enumerate(expertise_list):
                if exp.get("confidence", 1.0) < 0.8:
                    low_confidence_items.append({
                        "classification": {"category": exp.get("category")},
                        "confidence": exp.get("confidence"),
                        "justification": exp.get("justification", ""),
                        "index": i,
                        "type": "expertise"
                    })
        
        elif agent_type == "role_levels":
            role_list = classification_result.get("role_levels", [])
            for i, role in enumerate(role_list):
                if role.get("confidence", 1.0) < 0.8:
                    low_confidence_items.append({
                        "classification": {
                            "expertise": role.get("expertise"),
                            "level": role.get("level")
                        },
                        "confidence": role.get("confidence"),
                        "justification": role.get("justification", ""),
                        "index": i,
                        "type": "role_level"
                    })
        
        elif agent_type == "org_unit":
            org_list = classification_result.get("org_units", [])
            for i, org in enumerate(org_list):
                if org.get("confidence", 1.0) < 0.8:
                    low_confidence_items.append({
                        "classification": {"unit": org.get("unit")},
                        "confidence": org.get("confidence"),
                        "justification": org.get("justification", ""),
                        "index": i,
                        "type": "org_unit"
                    })
        
        return low_confidence_items
    
    def _apply_validation_result(self, classification_result: Dict, original_item: Dict, 
                               validation_response: Dict, agent_type: str) -> Dict:
        """Apply the validation result to the classification"""
        action = validation_response.get("action", "validate")
        validated_classification = validation_response.get("validated_classification", {})
        validation_reason = validation_response.get("validation_reason", "")
        final_confidence = validation_response.get("final_confidence", original_item["confidence"])
        
        # Find and update the item in the classification result
        if agent_type == "expertise":
            expertise_list = classification_result.get("expertise", [])
            index = original_item["index"]
            if 0 <= index < len(expertise_list):
                if action == "correct":
                    # Replace with corrected classification
                    expertise_list[index] = {
                        "category": validated_classification.get("category", expertise_list[index]["category"]),
                        "confidence": final_confidence,
                        "justification": validated_classification.get("justification", expertise_list[index]["justification"]),
                        "validation_applied": True,
                        "validation_action": action,
                        "validation_reason": validation_reason,
                        "original_confidence": original_item["confidence"]
                    }
                else:  # validate
                    # Keep original but update confidence and add validation info
                    expertise_list[index]["confidence"] = final_confidence
                    expertise_list[index]["validation_applied"] = True
                    expertise_list[index]["validation_action"] = action
                    expertise_list[index]["validation_reason"] = validation_reason
                    expertise_list[index]["original_confidence"] = original_item["confidence"]
        
        elif agent_type == "role_levels":
            role_list = classification_result.get("role_levels", [])
            index = original_item["index"]
            if 0 <= index < len(role_list):
                if action == "correct":
                    role_list[index] = {
                        "expertise": validated_classification.get("expertise", role_list[index]["expertise"]),
                        "level": validated_classification.get("level", role_list[index]["level"]),
                        "confidence": final_confidence,
                        "justification": validated_classification.get("justification", role_list[index]["justification"]),
                        "validation_applied": True,
                        "validation_action": action,
                        "validation_reason": validation_reason,
                        "original_confidence": original_item["confidence"]
                    }
                else:  # validate
                    role_list[index]["confidence"] = final_confidence
                    role_list[index]["validation_applied"] = True
                    role_list[index]["validation_action"] = action
                    role_list[index]["validation_reason"] = validation_reason
                    role_list[index]["original_confidence"] = original_item["confidence"]
        
        elif agent_type == "org_unit":
            org_list = classification_result.get("org_units", [])
            index = original_item["index"]
            if 0 <= index < len(org_list):
                if action == "correct":
                    org_list[index] = {
                        "unit": validated_classification.get("unit", org_list[index]["unit"]),
                        "confidence": final_confidence,
                        "justification": validated_classification.get("justification", org_list[index]["justification"]),
                        "validation_applied": True,
                        "validation_action": action,
                        "validation_reason": validation_reason,
                        "original_confidence": original_item["confidence"]
                    }
                else:  # validate
                    org_list[index]["confidence"] = final_confidence
                    org_list[index]["validation_applied"] = True
                    org_list[index]["validation_action"] = action
                    org_list[index]["validation_reason"] = validation_reason
                    org_list[index]["original_confidence"] = original_item["confidence"]
        
        return classification_result
    
    def _create_validation_feedback(self, resume_id: str, original_item: Dict, 
                                  validation_response: Dict, agent_type: str) -> Dict:
        """Create feedback for the feedback manager based on validation"""
        action = validation_response.get("action", "validate")
        validation_reason = validation_response.get("validation_reason", "")
        
        # Determine if this is positive or negative feedback
        rating = "positive" if action == "validate" else "negative"
        
        # Create structured feedback text
        if agent_type == "expertise":
            key = original_item["classification"]["category"]
            feedback_text = f"expertise :: {key} :: {validation_reason}"
        elif agent_type == "role_levels":
            expertise = original_item["classification"]["expertise"]
            level = original_item["classification"]["level"]
            key = f"{expertise}-{level}"
            feedback_text = f"role_level :: {key} :: {validation_reason}"
        elif agent_type == "org_unit":
            key = original_item["classification"]["unit"]
            feedback_text = f"org_unit :: {key} :: {validation_reason}"
        else:
            feedback_text = validation_reason
        
        return {
            "resume_id": resume_id,
            "rating": rating,
            "reason": feedback_text,
            "source": "validation_agent",  # Mark as agent feedback
            "original_confidence": original_item["confidence"],
            "final_confidence": validation_response.get("final_confidence"),
            "validation_action": action
        }
    
    def _parse_response(self, response_text):
        """Parse the validation response JSON"""
        try:
            response_text = clean_json_string(response_text)
            response_json = json.loads(response_text)
            return response_json
        except json.JSONDecodeError:
            logging.error(f"Failed to parse validation response: {response_text}")
            raise ValueError("Validation response is not valid JSON")
    
    def _validate_result(self, result):
        """Validate the structure of the validation response"""
        if not isinstance(result, dict):
            return False
        
        required_fields = ["action", "validated_classification", "validation_reason"]
        return all(field in result for field in required_fields)
    
    def _get_fallback_result(self, errors):
        """Generate fallback validation result"""
        return {
            "action": "validate",
            "validated_classification": {},
            "validation_reason": "Validation failed due to processing errors",
            "error": True,
            "errors": errors
        }


def clean_json_string(json_string):
    """Clean JSON string by removing markdown formatting"""
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip() 