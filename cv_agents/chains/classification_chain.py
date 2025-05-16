from ..expertise.agent import ExpertiseAgent
from ..role.agent import RoleLevelAgent
from ..org_unit.agent import OrgUnitAgent
from ..interpreter.agent import InterpreterAgent
from ..utils.data_extractor import extract_cv_sections
from ..utils.vector_utils import CVVectorizer
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

class CVClassificationOrchestrator:
    def __init__(self, memory_path: str = "memory/cv_classifications.json", custom_config: Optional[Dict[str, Any]] = None):
        self.custom_config = custom_config or {}
        
        # Initialize agents with custom configuration
        expertise_config = self.custom_config.get("expertise", {})
        role_config = self.custom_config.get("role_levels", {})
        org_unit_config = self.custom_config.get("org_units", {})
        
        self.expertise_agent = ExpertiseAgent(custom_config=expertise_config)
        self.role_level_agent = RoleLevelAgent(custom_config=role_config)
        self.org_unit_agent = OrgUnitAgent(custom_config=org_unit_config)
        
        self.memory_path = memory_path
        self.memory = self._load_memory()
        self.vectorizer = CVVectorizer()
    
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
            # Process with agents
            print("Processing expertise areas...")
            expertise_results = self.expertise_agent.process(cv_sections)
            
            print("Processing role levels...")
            role_input = {**cv_sections, "expertise_results": expertise_results}
            role_results = self.role_level_agent.process(role_input)
            
            print("Processing organizational unit...")
            org_input = {**cv_sections, "expertise_results": expertise_results, "role_results": role_results}
            org_results = self.org_unit_agent.process(org_input)
            
            # Combine results
            result = {
                "resume_id": cv_data.get("resume_id", ""),
                "timestamp": datetime.now().isoformat(),
                "source": "new_analysis",
                "expertise": expertise_results,
                "role_levels": role_results,
                "org_unit": org_results
            }
            
            # Store in memory
            self.memory["classifications"].append(result)
            self._save_memory()
            
            return result
            
        except Exception as e:
            print(f"Error processing CV: {str(e)}")
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
