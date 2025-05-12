from ..expertise.agent import ExpertiseAgent
from ..role.agent import RoleLevelAgent
from ..org_unit.agent import OrgUnitAgent
from ..utils.data_extractor import extract_cv_sections
from ..utils.vector_utils import CVVectorizer
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class CVClassificationOrchestrator:
    def __init__(self, memory_path: str = "memory/cv_classifications.json"):
        self.expertise_agent = ExpertiseAgent()
        self.role_level_agent = RoleLevelAgent()
        self.org_unit_agent = OrgUnitAgent()
        self.memory_path = memory_path
        self.memory = self._load_memory()
        self.vectorizer = CVVectorizer()
    
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
        
        print("DEBUG: checking for historical data")
        
        # First check exact match
        for classification in self.memory["classifications"]:
            if classification["resume_id"] == resume_id:
                return classification, []
        
        print("DEBUG: no exact match found")

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
        exact_match, similar_cvs = None, None # self._get_historical_data(cv_data)
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
