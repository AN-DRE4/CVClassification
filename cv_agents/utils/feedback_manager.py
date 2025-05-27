import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

class FeedbackManager:
    """Manages feedback storage and retrieval for CV classification agents"""
    
    def __init__(self, feedback_path: str = "cv_agents/data/feedback.json"):
        self.feedback_path = feedback_path
        self.feedback_data = self._load_feedback()
        
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    
    def _load_feedback(self) -> Dict:
        """Load feedback data from file"""
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading feedback data: {e}")
                return self._get_empty_feedback_structure()
        return self._get_empty_feedback_structure()
    
    def _get_empty_feedback_structure(self) -> Dict:
        """Get empty feedback data structure"""
        return {
            "expertise_feedback": [],
            "role_level_feedback": [],
            "org_unit_feedback": [],
            "general_feedback": [],
            "feedback_stats": {
                "total_positive": 0,
                "total_negative": 0,
                "last_updated": None
            }
        }
    
    def _save_feedback(self):
        """Save feedback data to file"""
        try:
            with open(self.feedback_path, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving feedback data: {e}")
    
    def add_feedback(self, resume_id: str, classification_result: Dict, user_feedback: Dict):
        """Add user feedback for a classification result"""
        feedback_entry = {
            "resume_id": resume_id,
            "timestamp": datetime.now().isoformat(),
            "rating": user_feedback.get("rating"),
            "reason": user_feedback.get("reason", ""),
            "classification_result": classification_result
        }
        
        # Add to general feedback
        self.feedback_data["general_feedback"].append(feedback_entry)
        
        # Add specific feedback for each agent type
        if "expertise" in classification_result:
            self._add_expertise_feedback(feedback_entry)
        
        if "role_levels" in classification_result:
            self._add_role_level_feedback(feedback_entry)
        
        if "org_unit" in classification_result:
            self._add_org_unit_feedback(feedback_entry)
        
        # Update stats
        if user_feedback.get("rating") == "positive":
            self.feedback_data["feedback_stats"]["total_positive"] += 1
        else:
            self.feedback_data["feedback_stats"]["total_negative"] += 1
        
        self.feedback_data["feedback_stats"]["last_updated"] = datetime.now().isoformat()
        
        # Save to file
        self._save_feedback()
    
    def _add_expertise_feedback(self, feedback_entry: Dict):
        """Add expertise-specific feedback"""
        expertise_results = feedback_entry["classification_result"]["expertise"]
        for exp in expertise_results.get("expertise", []):
            expertise_feedback = {
                "resume_id": feedback_entry["resume_id"],
                "timestamp": feedback_entry["timestamp"],
                "rating": feedback_entry["rating"],
                "reason": feedback_entry["reason"],
                "category": exp["category"],
                "confidence": exp["confidence"],
                "justification": exp.get("justification", "")
            }
            self.feedback_data["expertise_feedback"].append(expertise_feedback)
    
    def _add_role_level_feedback(self, feedback_entry: Dict):
        """Add role level-specific feedback"""
        role_results = feedback_entry["classification_result"]["role_levels"]
        for role in role_results.get("role_levels", []):
            role_feedback = {
                "resume_id": feedback_entry["resume_id"],
                "timestamp": feedback_entry["timestamp"],
                "rating": feedback_entry["rating"],
                "reason": feedback_entry["reason"],
                "expertise": role["expertise"],
                "level": role["level"],
                "confidence": role["confidence"],
                "justification": role.get("justification", "")
            }
            self.feedback_data["role_level_feedback"].append(role_feedback)
    
    def _add_org_unit_feedback(self, feedback_entry: Dict):
        """Add org unit-specific feedback"""
        org_results = feedback_entry["classification_result"]["org_unit"]
        for unit in org_results.get("org_units", []):
            org_feedback = {
                "resume_id": feedback_entry["resume_id"],
                "timestamp": feedback_entry["timestamp"],
                "rating": feedback_entry["rating"],
                "reason": feedback_entry["reason"],
                "unit": unit["unit"],
                "confidence": unit["confidence"],
                "justification": unit.get("justification", "")
            }
            self.feedback_data["org_unit_feedback"].append(org_feedback)
    
    def get_expertise_feedback(self, category: Optional[str] = None) -> List[Dict]:
        """Get expertise feedback, optionally filtered by category"""
        feedback = self.feedback_data["expertise_feedback"]
        if category:
            return [f for f in feedback if f["category"] == category]
        return feedback
    
    def get_role_level_feedback(self, expertise: Optional[str] = None, level: Optional[str] = None) -> List[Dict]:
        """Get role level feedback, optionally filtered by expertise and/or level"""
        feedback = self.feedback_data["role_level_feedback"]
        if expertise:
            feedback = [f for f in feedback if f["expertise"] == expertise]
        if level:
            feedback = [f for f in feedback if f["level"] == level]
        return feedback
    
    def get_org_unit_feedback(self, unit: Optional[str] = None) -> List[Dict]:
        """Get org unit feedback, optionally filtered by unit"""
        feedback = self.feedback_data["org_unit_feedback"]
        if unit:
            return [f for f in feedback if f["unit"] == unit]
        return feedback
    
    def get_feedback_summary(self, agent_type: str, item: str) -> Dict:
        """Get feedback summary for a specific item"""
        if agent_type == "expertise":
            feedback = self.get_expertise_feedback(item)
        elif agent_type == "role_level":
            feedback = self.get_role_level_feedback(expertise=item.split("_")[0], level=item.split("_")[1] if "_" in item else None)
        elif agent_type == "org_unit":
            feedback = self.get_org_unit_feedback(item)
        else:
            return {}
        
        if not feedback:
            return {"positive_count": 0, "negative_count": 0, "confidence_adjustment": 0.0}
        
        positive_count = len([f for f in feedback if f["rating"] == "positive"])
        negative_count = len([f for f in feedback if f["rating"] == "negative"])
        
        # Calculate confidence adjustment based on feedback ratio
        total_feedback = positive_count + negative_count
        if total_feedback > 0:
            positive_ratio = positive_count / total_feedback
            # Adjust confidence: +0.1 for mostly positive, -0.1 for mostly negative
            if positive_ratio > 0.7:
                confidence_adjustment = 0.1
            elif positive_ratio < 0.3:
                confidence_adjustment = -0.1
            else:
                confidence_adjustment = 0.0
        else:
            confidence_adjustment = 0.0
        
        return {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "confidence_adjustment": confidence_adjustment,
            "recent_feedback": feedback[-5:] if feedback else []  # Last 5 feedback entries
        }
    
    def clear_feedback(self):
        """Clear all feedback data"""
        self.feedback_data = self._get_empty_feedback_structure()
        self._save_feedback()
    
    def get_stats(self) -> Dict:
        """Get overall feedback statistics"""
        return self.feedback_data["feedback_stats"] 