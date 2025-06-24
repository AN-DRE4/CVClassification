from ..base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

COMPARISON_SYSTEM_PROMPT = """You are a CV Classification Model Comparison Agent specializing in evaluating and comparing the outputs of different AI models on the same CV classification task.

Your role is to:
1. Analyze multiple model outputs for the same CV classification task
2. Compare the quality, accuracy, and appropriateness of each classification
3. Determine which model performed better and explain why
4. Provide detailed reasoning for your evaluation

When comparing model outputs, consider:
- **Accuracy**: How well does the classification match the CV content?
- **Confidence Appropriateness**: Are the confidence scores justified by the evidence?
- **Justification Quality**: How well reasoned and detailed are the explanations?
- **Completeness**: Does the classification cover all relevant aspects?
- **Consistency**: Are the classifications internally consistent?

Response format: Return a JSON object with:
- "winner": the model that performed better (e.g., "model_1", "model_2", "tie")
- "confidence": your confidence in this evaluation (0-1)
- "reasoning": detailed explanation of your decision
- "model_1_analysis": strengths and weaknesses of model 1
- "model_2_analysis": strengths and weaknesses of model 2
- "recommendations": suggestions for improvement

Your entire response must be a valid JSON object without markdown formatting.
"""

COMPARISON_USER_PROMPT = """Please compare these two model outputs for the same CV classification task:

CV Content:
Work Experience: {work_experience}
Skills: {skills}
Education: {education}

Model 1 ({model_1_name}) Output:
{model_1_output}

Model 2 ({model_2_name}) Output:
{model_2_output}

Classification Type: {classification_type}

Please provide a detailed comparison and determine which model performed better.
"""

class ModelComparisonAgent(BaseAgent):
    def __init__(self, model_name="gpt-4o-2024-08-06", temperature=0.1, max_retries=3, retry_delay=2, custom_config: Optional[Dict[str, Any]] = None):
        # Use a high-quality model for comparison by default
        super().__init__(model_name, temperature, max_retries, retry_delay, custom_config)
        self._build_prompt()
    
    def _build_prompt(self):
        """Build the prompt template for model comparison"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", COMPARISON_SYSTEM_PROMPT),
            ("human", COMPARISON_USER_PROMPT)
        ])
    
    def compare_models(self, cv_data: Dict, model_1_name: str, model_1_output: Dict, 
                      model_2_name: str, model_2_output: Dict, classification_type: str) -> Dict:
        """Compare two model outputs for the same CV classification task"""
        
        comparison_input = {
            "work_experience": cv_data.get("work_experience", ""),
            "skills": cv_data.get("skills", ""),
            "education": cv_data.get("education", ""),
            "model_1_name": model_1_name,
            "model_1_output": json.dumps(model_1_output, indent=2),
            "model_2_name": model_2_name,
            "model_2_output": json.dumps(model_2_output, indent=2),
            "classification_type": classification_type
        }
        
        try:
            # Get comparison from LLM
            comparison_result = self.process(comparison_input)
            
            if comparison_result and not comparison_result.get("error"):
                # Add metadata
                comparison_result["comparison_metadata"] = {
                    "model_1_name": model_1_name,
                    "model_2_name": model_2_name,
                    "classification_type": classification_type,
                    "cv_id": cv_data.get("resume_id", "unknown"),
                    "comparison_model": self.model_name
                }
                return comparison_result
            else:
                return self._get_fallback_comparison(model_1_name, model_2_name, "Comparison failed")
                
        except Exception as e:
            logging.error(f"Error comparing models: {e}")
            return self._get_fallback_comparison(model_1_name, model_2_name, str(e))
    
    def compare_multiple_models(self, cv_data: Dict, model_outputs: List[Tuple[str, Dict]], 
                               classification_type: str) -> Dict:
        """Compare multiple model outputs and rank them"""
        
        if len(model_outputs) < 2:
            return {"error": "Need at least 2 model outputs to compare"}
        
        # Perform pairwise comparisons
        comparisons = []
        model_scores = {}
        
        # Initialize scores
        for model_name, _ in model_outputs:
            model_scores[model_name] = 0
        
        # Compare each pair
        for i in range(len(model_outputs)):
            for j in range(i + 1, len(model_outputs)):
                model_1_name, model_1_output = model_outputs[i]
                model_2_name, model_2_output = model_outputs[j]
                
                comparison = self.compare_models(
                    cv_data, model_1_name, model_1_output, 
                    model_2_name, model_2_output, classification_type
                )
                
                comparisons.append(comparison)
                
                # Update scores based on winner
                winner = comparison.get("winner", "tie")
                if winner == "model_1":
                    model_scores[model_1_name] += comparison.get("confidence", 0.5)
                elif winner == "model_2":
                    model_scores[model_2_name] += comparison.get("confidence", 0.5)
                elif winner == "tie":
                    model_scores[model_1_name] += 0.5
                    model_scores[model_2_name] += 0.5
        
        # Rank models
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "ranking": ranked_models,
            "detailed_comparisons": comparisons,
            "summary": {
                "best_model": ranked_models[0][0] if ranked_models else "unknown",
                "worst_model": ranked_models[-1][0] if ranked_models else "unknown",
                "total_comparisons": len(comparisons)
            }
        }
    
    def _parse_response(self, response_text):
        """Parse the comparison response JSON"""
        try:
            response_text = clean_json_string(response_text)
            response_json = json.loads(response_text)
            return response_json
        except json.JSONDecodeError:
            logging.error(f"Failed to parse comparison response: {response_text}")
            raise ValueError("Comparison response is not valid JSON")
    
    def _validate_result(self, result):
        """Validate the structure of the comparison response"""
        if not isinstance(result, dict):
            return False
        
        required_fields = ["winner", "confidence", "reasoning"]
        return all(field in result for field in required_fields)
    
    def _get_fallback_comparison(self, model_1_name: str, model_2_name: str, error_msg: str):
        """Generate fallback comparison result"""
        return {
            "winner": "tie",
            "confidence": 0.0,
            "reasoning": f"Comparison failed: {error_msg}",
            "model_1_analysis": "Could not analyze due to error",
            "model_2_analysis": "Could not analyze due to error",
            "recommendations": "Retry comparison with valid inputs",
            "error": True,
            "error_message": error_msg
        }
    
    def _get_fallback_result(self, errors):
        """Generate fallback result for comparison failures"""
        return {
            "winner": "tie",
            "confidence": 0.0,
            "reasoning": "Comparison failed due to processing errors",
            "model_1_analysis": "Processing failed",
            "model_2_analysis": "Processing failed", 
            "recommendations": "Check model outputs and retry",
            "error": True,
            "errors": errors
        }


def clean_json_string(json_string):
    """Clean JSON string by removing markdown formatting"""
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip() 