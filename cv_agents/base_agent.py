from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json
import time
import logging
from typing import Dict, Optional, Any

class BaseAgent:
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, retry_delay=2, custom_config: Optional[Dict[str, Any]] = None):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.prompt = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.custom_config = custom_config or {}
        
        # Initialize feedback manager
        from .utils.feedback_manager import FeedbackManager
        self.feedback_manager = FeedbackManager()
        
    def process(self, cv_data):
        """Process a CV with the agent with automatic retries and feedback integration"""
        if not self.prompt:
            raise NotImplementedError("Each agent must define its prompt")
        
        # Apply any custom configurations to the data
        self._apply_custom_config(cv_data)
        
        # Format the prompt with CV data
        formatted_prompt = self.prompt.format_prompt(**cv_data)
        # Initialize counters and tracking
        retries = 0
        result = None
        errors = []
        # Try processing with retries
        while retries <= self.max_retries:
            try:
                # Get response from LLM
                response = self.llm.invoke(formatted_prompt.to_messages())
                # Parse the response
                result = self._parse_response(response.content)
                # If we reach here, parsing was successful
                # For additional validation, check if the result is somewhat valid
                if self._validate_result(result):
                    # Apply feedback adjustments to the result
                    result = self._apply_feedback_adjustments(result)
                    return result
                else:
                    error_msg = f"Invalid result format after parsing: {result}"
                    logging.warning(error_msg)
                    errors.append(error_msg)
            except Exception as e:
                error_msg = f"Attempt {retries + 1} failed: {str(e)}"
                logging.warning(error_msg)
                errors.append(error_msg)
            
            # If we reach the max retries, break the loop
            if retries >= self.max_retries:
                break
                
            # Exponential backoff for retries
            sleep_time = self.retry_delay * (2 ** retries)
            logging.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            retries += 1
        
        # If we get here, all retries failed
        logging.error(f"All {self.max_retries} retry attempts failed")
        
        # Return a fallback result if we couldn't get a valid one
        return self._get_fallback_result(errors)
    
    def _apply_custom_config(self, cv_data):
        """Apply any custom configurations to the data before processing
        This is meant to be overridden by subclasses for specific customization needs"""
        pass
    
    def _apply_feedback_adjustments(self, result):
        """Apply feedback-based adjustments to the classification result
        This should be overridden by subclasses for agent-specific adjustments"""
        return result
    
    def _parse_response(self, response_text):
        """Parse the LLM response into structured data"""
        raise NotImplementedError("Each agent must implement parsing logic")
    
    def _validate_result(self, result):
        """Validate the parsed result to ensure it's in the expected format
        Subclasses can override this for more specific validation."""
        return isinstance(result, dict) and len(result) > 0
    
    def _get_fallback_result(self, errors):
        """Provide a fallback result when all retries fail
        Subclasses should override this for specific fallback responses."""
        return {
            "error": True,
            "message": "Failed to process after multiple retries",
            "details": errors
        }
        
    def update_config(self, new_config: Dict[str, Any]):
        """Update the agent's custom configuration"""
        self.custom_config.update(new_config)
        # Optionally rebuild prompts or other components when config changes
        self._on_config_updated()
        
    def _on_config_updated(self):
        """Hook called when configuration is updated
        Override in subclasses to rebuild prompt templates or other configuration-dependent components"""
        pass
    
    def get_feedback_context(self, agent_type: str) -> str:
        """Get feedback context to include in prompts for learning from past feedback"""
        feedback_stats = self.feedback_manager.get_stats()
        
        if feedback_stats["total_positive"] == 0 and feedback_stats["total_negative"] == 0:
            return ""
        
        # Use the new targeted feedback context method
        targeted_context = self.feedback_manager.get_targeted_feedback_context(agent_type)
        
        if targeted_context:
            # We have targeted feedback, use that
            return targeted_context
        else:
            # Fall back to general feedback context for backwards compatibility
            context = f"\n\nIMPORTANT: Based on user feedback from previous classifications:\n"
            context += f"- Total positive feedback: {feedback_stats['total_positive']}\n"
            context += f"- Total negative feedback: {feedback_stats['total_negative']}\n"
            
            # Add general feedback insights
            if agent_type == "expertise":
                recent_feedback = self.feedback_manager.feedback_data.get("expertise_feedback", [])[-10:]
            elif agent_type == "role_level":
                recent_feedback = self.feedback_manager.feedback_data.get("role_level_feedback", [])[-10:]
            elif agent_type == "org_unit":
                recent_feedback = self.feedback_manager.feedback_data.get("org_unit_feedback", [])[-10:]
            else:
                recent_feedback = []
            
            if recent_feedback:
                context += "\nRecent feedback patterns:\n"
                positive_feedback = [f for f in recent_feedback if f.get("rating") == "positive"]
                negative_feedback = [f for f in recent_feedback if f.get("rating") == "negative"]
                
                if positive_feedback:
                    context += "✓ Users appreciated classifications that:\n"
                    for feedback in positive_feedback[-3:]:  # Last 3 positive
                        if feedback.get("reason"):
                            context += f"  - {feedback['reason'][:100]}...\n"
                
                if negative_feedback:
                    context += "✗ Users found issues with classifications that:\n"
                    for feedback in negative_feedback[-3:]:  # Last 3 negative
                        if feedback.get("reason"):
                            context += f"  - {feedback['reason'][:100]}...\n"
                
                context += "\nPlease consider this feedback when making your classification.\n"
            
            return context
