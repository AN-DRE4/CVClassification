from ..expertise.agent import ExpertiseAgent
from ..role.agent import RoleLevelAgent
from ..org_unit.agent import OrgUnitAgent
from ..interpreter.agent import InterpreterAgent
from ..validation.agent import ValidationAgent
from ..comparison.agent import ModelComparisonAgent
from ..utils.data_extractor import extract_cv_sections
from ..utils.vector_utils import CVVectorizer
from ..utils.feedback_manager import FeedbackManager
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

class CVClassificationOrchestrator:
    def __init__(self, memory_path: str = "memory/cv_classifications.json", custom_config: Optional[Dict[str, Any]] = None, model_name: str = "gpt-4o-mini-2024-07-18"):
        self.custom_config = custom_config or {}
        self.model_name = model_name
        
        # Initialize agents with custom configuration and model
        expertise_config = self.custom_config.get("expertise", {})
        role_config = self.custom_config.get("role_levels", {})
        org_unit_config = self.custom_config.get("org_units", {})
        
        self.expertise_agent = ExpertiseAgent(model_name=model_name, custom_config=expertise_config)
        self.role_level_agent = RoleLevelAgent(model_name=model_name, custom_config=role_config)
        self.org_unit_agent = OrgUnitAgent(model_name=model_name, custom_config=org_unit_config)
        self.validation_agent = ValidationAgent(model_name=model_name)  # Add validation agent
        
        self.memory_path = memory_path
        self.memory = self._load_memory()
        self.vectorizer = CVVectorizer()
        
        # Initialize feedback manager
        self.feedback_manager = FeedbackManager()
        
        # Initialize comparison agent (uses best model for comparisons)
        self.comparison_agent = ModelComparisonAgent()
    
    def update_model(self, model_name: str):
        """Update the model used by all agents"""
        self.model_name = model_name
        self.expertise_agent.update_model(model_name)
        self.role_level_agent.update_model(model_name)
        self.org_unit_agent.update_model(model_name)
        self.validation_agent.update_model(model_name)
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update the configuration for the orchestrator and its agents"""
        self.custom_config.update(new_config)
        
        # Update agent configurations
        if "expertise" in new_config:
            self.expertise_agent.update_config(new_config["expertise"])
        
        if "role_levels" in new_config:
            self.role_level_agent.update_config(new_config["role_levels"])
        
        if "org_units" in new_config:
            self.org_unit_agent.update_config(new_config["org_units"])
    
    def load_config_from_file(self, config_path: str):
        """Load configuration from a JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.update_config(config)
                return True
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            return False
    
    def load_config_from_interpreter(self, file_path: str, interpretation_description: str, agent_type: str):
        """Load configuration using the interpreter agent"""
        try:
            result = InterpreterAgent.process_file_for_agent(
                file_path=file_path,
                interpretation_description=interpretation_description,
                agent_type=agent_type
            )
            
            if not result:
                print(f"Error: Interpreter couldn't process the file for {agent_type}")
                return False
            
            # Create a configuration dictionary for the appropriate agent type
            config = {}
            if agent_type == "expertise":
                config = {"expertise": result}
            elif agent_type == "role_levels":
                config = {"role_levels": result}
            elif agent_type == "org_units":
                config = {"org_units": result}
            else:
                print(f"Error: Unknown agent type {agent_type}")
                return False
            
            # Update the configuration
            self.update_config(config)
            return True
            
        except Exception as e:
            print(f"Error loading configuration using interpreter: {e}")
            return False
    
    def _load_memory(self) -> Dict:
        """Load classification memory from file"""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        return {"classifications": []}
    
    def _save_memory(self):
        """Save classification memory to file"""
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def _get_historical_data(self, cv_data: Dict) -> Tuple[Optional[Dict], List[str]]:
        """Retrieve historical classification data for similar CVs"""
        resume_id = cv_data.get("resume_id")
        if not resume_id:
            return None, []
        
        # First check exact match
        for classification in self.memory["classifications"]:
            if classification["resume_id"] == resume_id:
                return classification, []

        # Then check similar CVs
        similar_cv_ids = self.vectorizer.find_similar_cvs(cv_data, resume_id)
        print("DEBUG: similar CVs: ", similar_cv_ids)
        similar_classifications = []
        
        for cv_id in similar_cv_ids:
            for classification in self.memory["classifications"]:
                if classification["resume_id"] == cv_id:
                    similar_classifications.append(classification)
                    break
        
        return None, similar_classifications
    
    def _merge_classifications(self, classifications: List[Dict]) -> Dict:
        """Merge multiple classifications into a consensus"""
        if not classifications:
            print("DEBUG: no classifications to merge")
            return {}
        
        # Initialize consensus
        consensus = {
            "expertise": {},
            "role_levels": {},
            "org_unit": {}
        }
        
        print("DEBUG: consensus initialized")
        # print("DEBUG: classifications keys: ", classifications[0].get("expertise", {}).keys())
        # print("DEBUG: classifications: ", classifications)
        # Aggregate expertise
        for classification in classifications:
            for exp in classification.get("expertise", {}).get("expertise", []):
                category = exp.get("category")
                if category not in consensus["expertise"]:
                    consensus["expertise"][category] = {
                        "confidence": 0,
                        "count": 0
                    }
                consensus["expertise"][category]["confidence"] += exp.get("confidence", 0)
                consensus["expertise"][category]["count"] += 1

        print("DEBUG: expertise aggregated")
        
        # Calculate average confidence for expertise
        for category in consensus["expertise"]:
            consensus["expertise"][category]["confidence"] /= consensus["expertise"][category]["count"]

        print("DEBUG: expertise calculated")
        
        # Aggregate role levels
        for classification in classifications:
            for role in classification.get("role_levels", {}).get("role_levels", []):
                expertise = role.get("expertise")
                level = role.get("level")
                key = f"{expertise}_{level}"
                
                if key not in consensus["role_levels"]:
                    consensus["role_levels"][key] = {
                        "expertise": expertise,
                        "level": level,
                        "confidence": 0,
                        "count": 0
                    }
                consensus["role_levels"][key]["confidence"] += role.get("confidence", 0)
                consensus["role_levels"][key]["count"] += 1
        
        # Calculate average confidence for role levels
        for key in consensus["role_levels"]:
            consensus["role_levels"][key]["confidence"] /= consensus["role_levels"][key]["count"]

        print("DEBUG: role levels calculated")
        
        # Aggregate org units
        for classification in classifications:
            for org in classification.get("org_unit", {}).get("org_units", []):
                unit = org.get("unit")
                if unit not in consensus["org_unit"]:
                    consensus["org_unit"][unit] = {
                        "confidence": 0,
                        "count": 0
                    }
                consensus["org_unit"][unit]["confidence"] += org.get("confidence", 0)
                consensus["org_unit"][unit]["count"] += 1
        
        # Calculate average confidence for org units
        for unit in consensus["org_unit"]:
            consensus["org_unit"][unit]["confidence"] /= consensus["org_unit"][unit]["count"]

        print("DEBUG: org units calculated")
        
        # Convert to final format
        final_consensus = {
            "expertise": {
                "expertise": [
                    {
                        "category": category,
                        "confidence": data["confidence"]
                    }
                    for category, data in consensus["expertise"].items()
                ]
            },
            "role_levels": {
                "role_levels": [
                    {
                        "expertise": data["expertise"],
                        "level": data["level"],
                        "confidence": data["confidence"]
                }
                    for data in consensus["role_levels"].values()
                ]
            },
            "org_unit": {
                "org_units": [
                    {
                        "unit": unit,
                    "confidence": data["confidence"]
                }
                for unit, data in consensus["org_unit"].items()
                ]
            },
        }

        print("DEBUG: final consensus: ", final_consensus)
        return final_consensus
    
    def process_cv(self, cv_data: Dict) -> Dict:
        """Process a CV through the entire agent chain with memory support"""
        # Check for historical data
        print("DEBUG: checking for historical data")
        exact_match, similar_cvs = None, None 
        # exact_match, similar_cvs = self._get_historical_data(cv_data)
        print("DEBUG: extracted historical data")
        if exact_match:
            print(f"Found exact match for resume {cv_data.get('resume_id')}")
            return exact_match
        
        print("DEBUG: no exact match found")
        
        # If we have similar CVs, get consensus
        if similar_cvs:
            print(f"For cv: {cv_data.get('resume_id')} found {len(similar_cvs)} similar CVs")
            consensus = self._merge_classifications(similar_cvs)
            if consensus:
                print("Using consensus from similar CVs")
                return {
                    "resume_id": cv_data.get("resume_id", ""),
                    "timestamp": datetime.now().isoformat(),
                    "source": "consensus",
                    **consensus
                }
        
        print("DEBUG: no consensus found")

        # Extract CV sections
        print(f"DEBUG: {cv_data.get('resume_id', '')} has no historical data or similar CVs")
        cv_sections = extract_cv_sections(cv_data)

        print("DEBUG: cv_sections: ", cv_sections.keys())
        
        try:
            # Process with agents using conversational validation
            print("Processing expertise areas with conversational validation...")
            expertise_results = self.expertise_agent.process_with_validation(cv_sections, self.validation_agent, "expertise")
            
            print("Processing role levels with conversational validation...")
            role_input = {**cv_sections, "expertise_results": expertise_results}
            role_results = self.role_level_agent.process_with_validation(role_input, self.validation_agent, "role_levels")
            
            print("Processing organizational unit with conversational validation...")
            org_input = {**cv_sections, "expertise_results": expertise_results, "role_results": role_results}
            org_results = self.org_unit_agent.process_with_validation(org_input, self.validation_agent, "org_unit")
            
            # Combine results
            result = {
                "resume_id": cv_data.get("resume_id", ""),
                "timestamp": datetime.now().isoformat(),
                "source": "new_analysis",
                "expertise": expertise_results,
                "role_levels": role_results,
                "org_unit": org_results
            }
            
            # Collect conversational validation metadata from all agents
            validation_conversations = []
            total_iterations = 0
            
            for agent_type, agent_result in [("expertise", expertise_results), ("role_levels", role_results), ("org_unit", org_results)]:
                if agent_result.get("validation_conversation"):
                    conversation_data = agent_result["validation_conversation"]
                    conversation_data["agent_type"] = agent_type
                    validation_conversations.append(conversation_data)
                    total_iterations += conversation_data.get("iterations_completed", 1)
            
            if validation_conversations:
                print(f"Completed {len(validation_conversations)} validation conversations with {total_iterations} total iterations")
                result["validation_conversations"] = validation_conversations
                result["total_validation_iterations"] = total_iterations
                
                # Add summary of validation effectiveness
                satisfied_count = sum(1 for conv in validation_conversations if conv.get("validator_satisfied", False))
                result["validation_success_rate"] = satisfied_count / len(validation_conversations) if validation_conversations else 0
            
            # Store in memory
            self.memory["classifications"].append(result)
            self._save_memory()
            
            return result
            
        except Exception as e:
            print(f"Error processing CV in classification_chain.py: {str(e)}")
            return {
                "resume_id": cv_data.get("resume_id", ""),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_classification_history(self) -> List[Dict]:
        """Get all historical classifications"""
        return self.memory["classifications"]
    
    def clear_memory(self):
        """Clear the classification memory"""
        self.memory = {"classifications": []}
        self._save_memory()
        self.vectorizer.clear_cache()
        print("Memory cleared: ", self.memory)

    def add_feedback(self, classification_result: Dict):
        """Add user feedback to a classification result"""
        if "user_feedback" not in classification_result:
            return False
        
        resume_id = classification_result.get("resume_id", "")
        user_feedback = classification_result["user_feedback"]
        
        # Remove user_feedback from classification_result for storage
        clean_result = {k: v for k, v in classification_result.items() if k != "user_feedback"}
        
        # Add feedback using the feedback manager
        self.feedback_manager.add_feedback(resume_id, clean_result, user_feedback)
        
        # Find and update the classification in memory
        for classification in self.memory["classifications"]:
            if classification["resume_id"] == resume_id:
                classification["user_feedback"] = user_feedback
                self._save_memory()
                return True
        
        return False
    
    def get_feedback_stats(self):
        """Get feedback statistics"""
        return self.feedback_manager.get_stats()
    
    def clear_feedback(self):
        """Clear all feedback data"""
        self.feedback_manager.clear_feedback()
    
    def process_cv_with_multiple_models(self, cv_data: Dict, model_names: List[str]) -> Dict:
        """Process a CV with multiple models and compare results"""
        results = {}
        model_outputs = []
        
        # Store original model
        original_model = self.model_name
        
        try:
            # Process with each model
            for model_name in model_names:
                print(f"Processing with model: {model_name}")
                
                # Update all agents to use this model
                self.update_model(model_name)
                
                # Process CV with this model
                result = self.process_cv(cv_data)
                results[model_name] = result
                
                # Store for comparison
                model_outputs.append((model_name, result))
            
            # Compare results if we have multiple models
            if len(model_outputs) >= 2:
                print("Comparing model outputs...")
                
                # Compare expertise results
                expertise_comparison = self.comparison_agent.compare_models(
                    cv_data,
                    model_outputs[0][0], model_outputs[0][1].get("expertise", {}),
                    model_outputs[1][0], model_outputs[1][1].get("expertise", {}),
                    "expertise"
                )
                
                # Compare role level results
                role_comparison = self.comparison_agent.compare_models(
                    cv_data,
                    model_outputs[0][0], model_outputs[0][1].get("role_levels", {}),
                    model_outputs[1][0], model_outputs[1][1].get("role_levels", {}),
                    "role_levels"
                )
                
                # Compare org unit results
                org_comparison = self.comparison_agent.compare_models(
                    cv_data,
                    model_outputs[0][0], model_outputs[0][1].get("org_unit", {}),
                    model_outputs[1][0], model_outputs[1][1].get("org_unit", {}),
                    "org_unit"
                )
                
                # Combine comparison results
                comparison_results = {
                    "expertise_comparison": expertise_comparison,
                    "role_levels_comparison": role_comparison,
                    "org_unit_comparison": org_comparison
                }
                
                return {
                    "cv_id": cv_data.get("resume_id", ""),
                    "models_tested": model_names,
                    "individual_results": results,
                    "comparisons": comparison_results,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "cv_id": cv_data.get("resume_id", ""),
                    "models_tested": model_names,
                    "individual_results": results,
                    "timestamp": datetime.now().isoformat()
                }
                
        finally:
            # Restore original model
            self.update_model(original_model)
