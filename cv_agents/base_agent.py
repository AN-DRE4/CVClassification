from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json
import time
import logging
from typing import Dict, Optional, Any, List

class BaseAgent:
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, retry_delay=2, custom_config: Optional[Dict[str, Any]] = None, max_validation_iterations: int = 3):
        self.model_name = model_name  # Store model name for reference
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
        
        # Configuration for conversational validation - can be overridden by custom_config
        default_max_iterations = max_validation_iterations
        self.max_validation_iterations = self.custom_config.get("max_validation_iterations", default_max_iterations)
        
        self.confidence_thresholds = {
            'very_high': 0.95,  # 95%+ confidence
            'high': 0.9,      # 90%-95% confidence
            'medium_high': 0.8,  # 80-90% confidence
            'medium': 0.6,    # 60-80% confidence
            'low': 0.4,       # 40-60% confidence
            'very_low': 0.0   # Below 40% confidence
        }
        
    def process_with_validation(self, cv_data, validation_agent, agent_type: str):
        """Process CV data with iterative validation until validator is satisfied"""
        current_classification = None
        iteration_count = 0
        validation_history = []
        
        # Track original confidence values from first iteration when each key first appears
        original_confidence_tracker = {}
        
        while iteration_count < self.max_validation_iterations:
            print(f"Starting {agent_type} classification iteration {iteration_count + 1}")
            
            # Prepare input for this iteration
            iteration_input = cv_data.copy()
            
            # If this is not the first iteration, include validation feedback
            if current_classification and validation_history:
                latest_feedback = validation_history[-1]
                iteration_input['validation_feedback'] = latest_feedback
                iteration_input['previous_classification'] = current_classification
                iteration_input['iteration'] = iteration_count + 1
            
            # Get classification from this agent
            current_classification = self.process(iteration_input)

            # print("DEBUG: got here 3", current_classification)
            
            if current_classification.get("error"):
                print(f"Error in {agent_type} agent classification: {current_classification}")
                break
            
            # Track original confidence values from first appearance
            self._track_original_confidence(current_classification, original_confidence_tracker, agent_type, iteration_count)
            
            # Gather available categories for this agent type
            available_categories = self._get_available_categories_for_validation(agent_type)
            
            # Send classification to validation agent for review
            validation_feedback = validation_agent.provide_feedback(
                cv_data, agent_type, current_classification, iteration_count + 1, available_categories
            )
            
            validation_history.append(validation_feedback)
            
            # Check if validator is satisfied
            if validation_feedback.get("validator_satisfied", False):
                print(f"{agent_type.capitalize()} agent classification approved by validator after {iteration_count + 1} iterations")
                break
            
            print(f"Validator feedback for {agent_type}: {validation_feedback.get('feedback_summary', 'No summary provided')}")
            # print(f"DEBUG: Validatior detailed feedback for {agent_type}: {validation_feedback.get('detailed_feedback', {})}")
            iteration_count += 1
        
        # Apply original confidence values to final result
        if current_classification and not current_classification.get("error"):
            self._apply_original_confidence_to_result(current_classification, original_confidence_tracker, agent_type)
        
        # Add conversation metadata to final result
        # print("DEBUG: got here 4: ", current_classification)
        # print("DEBUG: got here 4.1: ", validation_history[-1].get("validator_satisfied", False))
        
        if current_classification and not current_classification.get("error"):
            current_classification["validation_conversation"] = {
                "iterations_completed": iteration_count + 1,
                "validator_satisfied": validation_history[-1].get("validator_satisfied", False) if validation_history else False,
                "conversation_history": validation_history,
                "max_iterations_reached": iteration_count >= self.max_validation_iterations
            }
            # print("DEBUG: got here 4.2: ", current_classification)
            
            # Generate automatic feedback for learning if the conversation was useful
            """if iteration_count > 0:  # Only if there was actual conversation
                self._generate_conversation_feedback(cv_data.get("resume_id", "unknown"), current_classification, validation_history, agent_type)"""

        # print("DEBUG: got here 5")

        return current_classification

    def _track_original_confidence(self, classification, tracker, agent_type, iteration_count):
        """Track original confidence values from first iteration when each key first appears"""
        if agent_type == "expertise":
            expertise_list = classification.get("expertise", [])
            for exp in expertise_list:
                key = exp.get("category")
                if key and key not in tracker:
                    tracker[key] = {
                        "original_confidence": exp.get("confidence", 0),
                        "first_iteration": iteration_count
                    }
        
        elif agent_type == "role_levels":
            role_levels = classification.get("role_levels", [])
            for role in role_levels:
                expertise = role.get("expertise", "")
                level = role.get("level", "")
                # Create unique key for expertise-level combination
                key = f"{expertise}_{level}"
                # Also track by just expertise for cases where level changes
                expertise_key = expertise
                
                if key and key not in tracker:
                    tracker[key] = {
                        "original_confidence": role.get("confidence", 0),
                        "original_level": level,
                        "expertise": expertise,
                        "first_iteration": iteration_count
                    }
                
                # Also track the first time we see this expertise area (regardless of level)
                if expertise_key and f"expertise_{expertise_key}" not in tracker:
                    tracker[f"expertise_{expertise_key}"] = {
                        "first_confidence": role.get("confidence", 0),
                        "first_level": level,
                        "first_iteration": iteration_count
                    }
        
        elif agent_type == "org_unit":
            org_units = classification.get("org_units", [])
            for unit in org_units:
                key = unit.get("unit")
                if key and key not in tracker:
                    tracker[key] = {
                        "original_confidence": unit.get("confidence", 0),
                        "first_iteration": iteration_count
                    }

    def _apply_original_confidence_to_result(self, classification, tracker, agent_type):
        """Apply the tracked original confidence values to the final result"""
        if agent_type == "expertise":
            expertise_list = classification.get("expertise", [])
            for exp in expertise_list:
                key = exp.get("category")
                if key in tracker:
                    exp["original_confidence"] = tracker[key]["original_confidence"]
        
        elif agent_type == "role_levels":
            role_levels = classification.get("role_levels", [])
            for role in role_levels:
                expertise = role.get("expertise", "")
                level = role.get("level", "")
                key = f"{expertise}_{level}"
                expertise_key = expertise
                
                # First try to find exact expertise-level match
                if key in tracker:
                    role["original_confidence"] = tracker[key]["original_confidence"]
                    role["original_level"] = tracker[key]["original_level"]
                else:
                    # If exact match not found, check if we have data for this expertise area
                    # This handles the case where level changed but expertise stayed the same
                    expertise_tracker_key = f"expertise_{expertise_key}"
                    if expertise_tracker_key in tracker:
                        role["original_confidence"] = tracker[expertise_tracker_key]["first_confidence"]
                        role["original_level"] = tracker[expertise_tracker_key]["first_level"]
                    else:
                        # Fallback: look for any tracker entry with the same expertise
                        print("DEBUG: didn't find exact match, looking for any tracker entry with the same expertise")
                        for tracker_key, tracker_data in tracker.items():
                            if tracker_data.get("expertise") == expertise:
                                role["original_confidence"] = tracker_data["original_confidence"]
                                role["original_level"] = tracker_data.get("original_level", level)
                                break
        
        elif agent_type == "org_unit":
            org_units = classification.get("org_units", [])
            for unit in org_units:
                key = unit.get("unit")
                if key in tracker:
                    unit["original_confidence"] = tracker[key]["original_confidence"]

    def _generate_conversation_feedback(self, resume_id: str, final_classification: Dict, validation_history: List[Dict], agent_type: str):
        """Generate automatic feedback based on the validation conversation"""
        if not validation_history:
            return
        
        last_feedback = validation_history[-1]
        was_satisfied = last_feedback.get("validator_satisfied", False)

        # print("DEBUG: got here 6: ", was_satisfied)
        
        # Create feedback entry
        feedback_entry = {
            "resume_id": resume_id,
            "rating": "positive" if was_satisfied else "negative",
            "source": "validation_conversation",
            "conversation_iterations": len(validation_history),
            "validator_satisfied": was_satisfied,
            "improvements_made": len(validation_history) > 1
        }
        
        # Add specific feedback for each classification item
        reason_parts = []
        
        if agent_type == "expertise":
            for exp in final_classification.get("expertise", []):
                category = exp.get("category")
                confidence = exp.get("confidence", 0)
                if was_satisfied:
                    reason_parts.append(f"expertise :: {category} :: Validated through conversation (final confidence: {confidence:.2f})")
                else:
                    reason_parts.append(f"expertise :: {category} :: Could not reach validation consensus (final confidence: {confidence:.2f})")
            # print("DEBUG: got here 7: ", reason_parts)

        elif agent_type == "role_levels":
            for role in final_classification.get("role_levels", []):
                expertise = role.get("expertise", "")
                level = role.get("level", "")
                confidence = role.get("confidence", 0)
                key = f"{expertise}-{level}"
                if was_satisfied:
                    reason_parts.append(f"role_level :: {key} :: Validated through conversation (final confidence: {confidence:.2f})")
                else:
                    reason_parts.append(f"role_level :: {key} :: Could not reach validation consensus (final confidence: {confidence:.2f})")
        
        elif agent_type == "org_unit":
            for unit in final_classification.get("org_units", []):
                unit_name = unit.get("unit", "")
                confidence = unit.get("confidence", 0)
                if was_satisfied:
                    reason_parts.append(f"org_unit :: {unit_name} :: Validated through conversation (final confidence: {confidence:.2f})")
                else:
                    reason_parts.append(f"org_unit :: {unit_name} :: Could not reach validation consensus (final confidence: {confidence:.2f})")
        
        # print("DEBUG: got here 8: ", reason_parts)
        # print("DEBUG: got here 8.1: ", final_classification)
        # print("DEBUG: got here 8.2: ", feedback_entry)
        
        if reason_parts:
            feedback_entry["reason"] = "\n".join(reason_parts)
            
            # Add the feedback to the feedback manager
            self.feedback_manager.add_feedback(
                resume_id,
                final_classification,
                feedback_entry
            )

        # print("DEBUG: got here 9: ", feedback_entry)

    def _get_confidence_tier(self, confidence: float) -> str: # TODO: eliminar isto
        """Get confidence tier based on improved granularity"""
        if confidence >= self.confidence_thresholds['very_high']:
            return 'very_high'
        elif confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['medium_high']:
            return 'medium_high'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        elif confidence >= self.confidence_thresholds['low']:
            return 'low'
        else:
            return 'very_low'
    
    def _adjust_confidence_scoring(self, initial_confidence: float, evidence_strength: str = "medium") -> float: # TODO: eliminar isto
        """Adjust confidence scoring with improved granularity"""
        # Map evidence strength to confidence adjustments
        evidence_adjustments = {
            "very_strong": 0.95,  # 95% confidence for very strong evidence
            "strong": 0.85,       # 85% confidence for strong evidence
            "medium": 0.70,       # 70% confidence for medium evidence
            "weak": 0.50,         # 50% confidence for weak evidence
            "very_weak": 0.30     # 30% confidence for very weak evidence
        }
        
        # Get base confidence from evidence
        base_confidence = evidence_adjustments.get(evidence_strength, initial_confidence)
        
        # Apply some randomness/adjustment based on specific factors
        # This can be overridden by subclasses for more specific logic
        adjusted = min(max(base_confidence, 0.1), 0.95)  # Keep between 10% and 95%
        
        return adjusted

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
                    # print(f"Result before feedback adjustments: {result}")
                    result = self._apply_feedback_adjustments(result)

                    # print("DEBUG: got here 1")
                    
                    # Apply improved confidence scoring
                    # result = self._apply_improved_confidence_scoring(result)
                    
                    # print("DEBUG: got here 2")
                    
                    # print("DEBUG: Result before return: ", result)
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
    
    '''def _apply_improved_confidence_scoring(self, result):
        """Apply improved confidence scoring with better granularity
        This should be overridden by subclasses for agent-specific improvements"""
        return result'''
    
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
        
        # Update max_validation_iterations if provided in new_config
        if "max_validation_iterations" in new_config:
            self.max_validation_iterations = new_config["max_validation_iterations"]
        
        # Optionally rebuild prompts or other components when config changes
        self._on_config_updated()
    
    def update_model(self, model_name: str):
        """Update the model used by this agent"""
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=self.llm.temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        # Rebuild prompts if they include model-specific context
        self._on_config_updated()
        
    def _on_config_updated(self):
        """Hook called when configuration is updated
        Override in subclasses to rebuild prompt templates or other configuration-dependent components"""
        pass
    
    def _get_available_categories_for_validation(self, agent_type: str) -> Dict[str, Any]:
        """Get available categories for validation based on agent type"""
        categories = {}
        
        if agent_type == "expertise":
            from cv_agents.expertise.agent import DEFAULT_EXPERTISE_CATEGORIES
            categories["expertise_categories"] = self.custom_config.get("expertise_categories", DEFAULT_EXPERTISE_CATEGORIES)
        
        elif agent_type == "role_levels":
            from cv_agents.role.agent import DEFAULT_ROLE_LEVELS
            from cv_agents.expertise.agent import DEFAULT_EXPERTISE_CATEGORIES
            categories["role_levels"] = self.custom_config.get("role_levels", DEFAULT_ROLE_LEVELS)
            categories["expertise_categories"] = self.custom_config.get("expertise_categories", DEFAULT_EXPERTISE_CATEGORIES)
        
        elif agent_type == "org_unit":
            from cv_agents.org_unit.agent import DEFAULT_ORG_UNITS
            categories["org_units"] = self.custom_config.get("org_units", DEFAULT_ORG_UNITS)
        
        return categories
    
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
