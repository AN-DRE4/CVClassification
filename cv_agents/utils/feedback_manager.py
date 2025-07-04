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
    
    def parse_structured_feedback(self, feedback_text: str) -> List[Dict[str, str]]:
        """Parse structured feedback in format 'area; key; feedback'
        
        Args:
            feedback_text: Multi-line text with each line in format "area; key; feedback"
            
        Returns:
            List of parsed feedback entries
        """
        parsed_feedback = []
        
        if not feedback_text or not feedback_text.strip():
            return parsed_feedback
        
        lines = feedback_text.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            # Split by semicolon
            parts = line.split('::')
            
            if len(parts) != 3:
                logging.warning(f"Invalid feedback format on line {line_num}: '{line}'. Expected format: 'area :: key :: feedback'")
                continue
            
            area = parts[0].strip().lower()
            key = parts[1].strip()
            feedback = parts[2].strip()
            
            # Validate area
            valid_areas = ['expertise', 'role_level', 'role_levels', 'org_unit', 'org_units', 'organizational_unit']
            if area not in valid_areas:
                logging.warning(f"Invalid area '{area}' on line {line_num}. Valid areas: {valid_areas}")
                continue
            
            # Normalize area names
            if area in ['role_level', 'role_levels']:
                area = 'role_level'
            elif area in ['org_unit', 'org_units', 'organizational_unit']:
                area = 'org_unit'
            
            parsed_feedback.append({
                'area': area,
                'key': key,
                'feedback': feedback,
                'line_number': line_num
            })
        
        return parsed_feedback
    
    def add_feedback(self, resume_id: str, classification_result: Dict, user_feedback: Dict):
        """Add user feedback for a classification result with new structured format"""
        feedback_text = user_feedback.get("reason", "")
        rating = user_feedback.get("rating", "neutral")
        feedback_source = user_feedback.get("source", "human")  # New: track feedback source

        print("DEBUG: got here 10: ", feedback_text)
        print("DEBUG: got here 10.1: ", rating)
        print("DEBUG: got here 10.2: ", feedback_source)
        
        # Try to parse structured feedback first
        parsed_feedback = self.parse_structured_feedback(feedback_text)
        
        if parsed_feedback:
            # Handle structured feedback
            self._add_structured_feedback(resume_id, classification_result, parsed_feedback, rating, feedback_source)
        else:
            # Fall back to old general feedback approach for backwards compatibility
            self._add_general_feedback(resume_id, classification_result, user_feedback)
        
        # Update overall stats with source tracking
        if rating == "positive":
            self.feedback_data["feedback_stats"]["total_positive"] += 1
            if feedback_source == "validation_agent":
                self.feedback_data["feedback_stats"]["agent_positive"] = self.feedback_data["feedback_stats"].get("agent_positive", 0) + 1
            else:
                self.feedback_data["feedback_stats"]["human_positive"] = self.feedback_data["feedback_stats"].get("human_positive", 0) + 1
        else:
            self.feedback_data["feedback_stats"]["total_negative"] += 1
            if feedback_source == "validation_agent":
                self.feedback_data["feedback_stats"]["agent_negative"] = self.feedback_data["feedback_stats"].get("agent_negative", 0) + 1
            else:
                self.feedback_data["feedback_stats"]["human_negative"] = self.feedback_data["feedback_stats"].get("human_negative", 0) + 1

        self.feedback_data["feedback_stats"]["last_updated"] = datetime.now().isoformat()
        
        # Save to file
        self._save_feedback()
    
    def add_validation_feedback(self, resume_id: str, validation_feedback_list: List[Dict]):
        """Add feedback from validation agent"""
        for validation_feedback in validation_feedback_list:
            # Convert validation feedback to user feedback format

            area = validation_feedback.get("reason").split("::")[0].strip()
            key = validation_feedback.get("reason").split("::")[1].strip()
            feedback = validation_feedback.get("reason").split("::")[2].strip()

            user_feedback = {
                "rating": validation_feedback.get("rating"),
                "reason": validation_feedback.get("reason"),
                "source": "validation_agent"
            }

            print("DEBUG: validation_feedback_list", validation_feedback_list)
            
            # Create a dummy classification result for the feedback
            if area == "expertise":
                classification_result = {
                    "expertise": {"expertise": [{
                        "category": key,
                        "confidence": validation_feedback.get("original_confidence"),
                        "justification": feedback
                    }]},
                    "role_levels": {"role_levels": []},
                    "org_unit": {"org_units": []}
                }
            elif area == "role_level":
                classification_result = {
                    "expertise": {"expertise": []},
                    "role_levels": {"role_levels": [{
                        "expertise": key.split("-")[0],
                        "level": key.split("-")[1],
                        "confidence": validation_feedback.get("original_confidence"),
                        "justification": feedback
                    }]},
                    "org_unit": {"org_units": []}
                }
            elif area == "org_unit":
                classification_result = {
                    "expertise": {"expertise": []},
                    "role_levels": {"role_levels": []},
                    "org_unit": {"org_units": [{
                        "unit": key,
                        "confidence": validation_feedback.get("original_confidence"),
                        "justification": feedback
                    }]}
                }
            else:
                classification_result = {
                    "expertise": {"expertise": []},
                    "role_levels": {"role_levels": []},
                    "org_unit": {"org_units": []}
                }           
            
            # Add the feedback
            self.add_feedback(resume_id, classification_result, user_feedback)
    
    def _add_structured_feedback(self, resume_id: str, classification_result: Dict, parsed_feedback: List[Dict], rating: str, source: str = "human"):
        """Add structured feedback entries"""
        timestamp = datetime.now().isoformat()
        
        for feedback_entry in parsed_feedback:
            area = feedback_entry['area']
            key = feedback_entry['key']
            feedback = feedback_entry['feedback']
            
            # Create feedback record
            feedback_record = {
                "resume_id": resume_id,
                "timestamp": timestamp,
                "rating": rating,
                "reason": feedback,
                "key": key,
                "line_number": feedback_entry.get('line_number', 0),
                "source": source  # Track feedback source
            }
            
            # Add classification context based on area
            if area == 'expertise':
                self._add_expertise_specific_feedback(feedback_record, classification_result, key)
            elif area == 'role_level':
                self._add_role_level_specific_feedback(feedback_record, classification_result, key)
            elif area == 'org_unit':
                self._add_org_unit_specific_feedback(feedback_record, classification_result, key)
    
    def _add_expertise_specific_feedback(self, feedback_record: Dict, classification_result: Dict, key: str):
        """Add expertise-specific structured feedback"""
        expertise_results = classification_result.get("expertise", {})
        
        # Find matching expertise category
        matching_expertise = None
        for exp in expertise_results.get("expertise", []):
            if exp.get("category", "").lower() == key.lower():
                matching_expertise = exp
                break
        
        if matching_expertise:
            feedback_record.update({
                "category": matching_expertise["category"],
                "confidence": matching_expertise["confidence"],
                "justification": matching_expertise.get("justification", "")
            })
        else:
            # Key not found in results, still store the feedback for learning
            feedback_record.update({
                "category": key,
                "confidence": None,
                "justification": f"Key '{key}' not found in classification results"
            })
        
        self.feedback_data["expertise_feedback"].append(feedback_record)
    
    def _add_role_level_specific_feedback(self, feedback_record: Dict, classification_result: Dict, key: str):
        """Add role level-specific structured feedback"""
        role_results = classification_result.get("role_levels", {})
        
        # Find matching role level (key could be "expertise-level" format or just expertise)
        matching_role = None
        for role in role_results.get("role_levels", []):
            expertise = role.get("expertise", "")
            level = role.get("level", "")
            
            # Try exact match with "expertise-level" format
            if f"{expertise}-{level}".lower() == key.lower():
                matching_role = role
                break
            # Try match with just expertise
            elif expertise.lower() == key.lower():
                matching_role = role
                break
            # Try match with just level
            elif level.lower() == key.lower():
                matching_role = role
                break
        
        if matching_role:
            feedback_record.update({
                "expertise": matching_role["expertise"],
                "level": matching_role["level"],
                "confidence": matching_role["confidence"],
                "justification": matching_role.get("justification", "")
            })
        else:
            # Key not found in results, still store the feedback for learning
            feedback_record.update({
                "expertise": key,
                "level": "unknown",
                "confidence": None,
                "justification": f"Key '{key}' not found in role level results"
            })
        
        self.feedback_data["role_level_feedback"].append(feedback_record)
    
    def _add_org_unit_specific_feedback(self, feedback_record: Dict, classification_result: Dict, key: str):
        """Add org unit-specific structured feedback"""
        org_results = classification_result.get("org_unit", {})
        
        # Find matching org unit
        matching_unit = None
        for unit in org_results.get("org_units", []):
            if unit.get("unit", "").lower() == key.lower():
                matching_unit = unit
                break
        
        if matching_unit:
            feedback_record.update({
                "unit": matching_unit["unit"],
                "confidence": matching_unit["confidence"],
                "justification": matching_unit.get("justification", "")
            })
        else:
            # Key not found in results, still store the feedback for learning
            feedback_record.update({
                "unit": key,
                "confidence": None,
                "justification": f"Key '{key}' not found in org unit results"
            })
        
        self.feedback_data["org_unit_feedback"].append(feedback_record)
        
    def _add_general_feedback(self, resume_id: str, classification_result: Dict, user_feedback: Dict):
        """Add general feedback for backwards compatibility"""
        feedback_entry = {
            "resume_id": resume_id,
            "timestamp": datetime.now().isoformat(),
            "rating": user_feedback.get("rating"),
            "reason": user_feedback.get("reason", ""),
            "classification_result": classification_result
        }
        
        # Add to general feedback
        self.feedback_data["general_feedback"].append(feedback_entry)
        
        # Add broad feedback for all classifications (backwards compatibility)
        if "expertise" in classification_result:
            self._add_expertise_feedback(feedback_entry)
        
        if "role_levels" in classification_result:
            self._add_role_level_feedback(feedback_entry)
        
        if "org_unit" in classification_result:
            self._add_org_unit_feedback(feedback_entry)
    
    def _add_expertise_feedback(self, feedback_entry: Dict):
        """Add expertise-specific feedback (backwards compatibility)"""
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
        """Add role level-specific feedback (backwards compatibility)"""
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
        """Add org unit-specific feedback (backwards compatibility)"""
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
            return [f for f in feedback if f.get("category", "").lower() == category.lower()]
        return feedback
    
    def get_role_level_feedback(self, expertise: Optional[str] = None, level: Optional[str] = None) -> List[Dict]:
        """Get role level feedback, optionally filtered by expertise and/or level"""
        feedback = self.feedback_data["role_level_feedback"]
        if expertise:
            feedback = [f for f in feedback if f.get("expertise", "").lower() == expertise.lower()]
        if level:
            feedback = [f for f in feedback if f.get("level", "").lower() == level.lower()]
        return feedback
    
    def get_org_unit_feedback(self, unit: Optional[str] = None) -> List[Dict]:
        """Get org unit feedback, optionally filtered by unit"""
        feedback = self.feedback_data["org_unit_feedback"]
        if unit:
            return [f for f in feedback if f.get("unit", "").lower() == unit.lower()]
        return feedback
    
    def get_feedback_summary(self, agent_type: str, item: str) -> Dict:
        """Get feedback summary for a specific item with enhanced targeted feedback support"""
        if agent_type == "expertise":
            # Get all feedback for this specific expertise category
            specific_feedback = [f for f in self.feedback_data["expertise_feedback"] 
                               if f.get("category", "").lower() == item.lower()]
        elif agent_type == "role_level":
            # Handle different formats for role level keys
            if "_" in item:
                expertise, level = item.split("_", 1)
                specific_feedback = [f for f in self.feedback_data["role_level_feedback"] 
                                   if (f.get("expertise", "").lower() == expertise.lower() and 
                                       f.get("level", "").lower() == level.lower())]
            else:
                # Just expertise provided
                specific_feedback = [f for f in self.feedback_data["role_level_feedback"] 
                                   if f.get("expertise", "").lower() == item.lower()]
        elif agent_type == "org_unit":
            specific_feedback = [f for f in self.feedback_data["org_unit_feedback"] 
                               if f.get("unit", "").lower() == item.lower()]
        else:
            return {"positive_count": 0, "negative_count": 0, "confidence_adjustment": 0.0}
        
        if not specific_feedback:
            return {"positive_count": 0, "negative_count": 0, "confidence_adjustment": 0.0}
        
        # Count positive/negative feedback specifically targeting this item
        positive_count = len([f for f in specific_feedback if f.get("rating") == "positive"])
        negative_count = len([f for f in specific_feedback if f.get("rating") == "negative"])
        
        # Calculate confidence adjustment based on targeted feedback
        total_feedback = positive_count + negative_count
        if total_feedback > 0:
            positive_ratio = positive_count / total_feedback
            
            # More sophisticated adjustment based on amount and ratio of feedback
            base_adjustment = 0.05  # Base adjustment amount
            
            # Scale adjustment based on amount of feedback (more feedback = more weight)
            feedback_weight = min(1.0, total_feedback / 5.0)  # Max weight at 5+ feedback entries
            
            if positive_ratio > 0.75:
                # Strongly positive feedback
                confidence_adjustment = base_adjustment * 2 * feedback_weight
            elif positive_ratio > 0.6:
                # Moderately positive feedback
                confidence_adjustment = base_adjustment * feedback_weight
            elif positive_ratio < 0.25:
                # Strongly negative feedback
                confidence_adjustment = -base_adjustment * 2 * feedback_weight
            elif positive_ratio < 0.4:
                # Moderately negative feedback
                confidence_adjustment = -base_adjustment * feedback_weight
            else:
                # Mixed feedback - no adjustment
                confidence_adjustment = 0.0
        else:
            confidence_adjustment = 0.0
        
        # Get recent targeted feedback for context
        recent_feedback = sorted(specific_feedback, key=lambda x: x.get("timestamp", ""))[-5:]
        
        return {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "confidence_adjustment": confidence_adjustment,
            "recent_feedback": recent_feedback,
            "feedback_strength": total_feedback  # How much feedback we have
        }
    
    def get_targeted_feedback_context(self, agent_type: str) -> str:
        """Get feedback context specifically for targeted feedback to include in agent prompts"""
        if agent_type == "expertise":
            feedback_list = self.feedback_data["expertise_feedback"]
            key_field = "category"
        elif agent_type == "role_level":
            feedback_list = self.feedback_data["role_level_feedback"]
            key_field = "expertise"
        elif agent_type == "org_unit":
            feedback_list = self.feedback_data["org_unit_feedback"]
            key_field = "unit"
        else:
            return ""
        
        if not feedback_list:
            return ""
        
        # Get recent targeted feedback (last 10)
        recent_feedback = sorted(feedback_list, key=lambda x: x.get("timestamp", ""))[-10:]
        
        if not recent_feedback:
            return ""
        
        context = "\n\nIMPORTANT: User feedback on specific classifications:\n"
        
        # Group feedback by key for better organization
        feedback_by_key = {}
        for feedback in recent_feedback:
            key = feedback.get(key_field, "unknown")
            if key not in feedback_by_key:
                feedback_by_key[key] = {"positive": [], "negative": []}
            
            if feedback.get("rating") == "positive":
                feedback_by_key[key]["positive"].append(feedback.get("reason", ""))
            else:
                feedback_by_key[key]["negative"].append(feedback.get("reason", ""))
        
        # Format the feedback context
        for key, feedback_data in feedback_by_key.items():
            if feedback_data["positive"] or feedback_data["negative"]:
                context += f"\nFor '{key}':\n"
                
                if feedback_data["positive"]:
                    context += "  ✓ Positive feedback:\n"
                    for pos_feedback in feedback_data["positive"][-2:]:  # Last 2 positive
                        context += f"    - {pos_feedback[:100]}...\n"
                
                if feedback_data["negative"]:
                    context += "  ✗ Negative feedback:\n"
                    for neg_feedback in feedback_data["negative"][-2:]:  # Last 2 negative
                        context += f"    - {neg_feedback[:100]}...\n"
        
        context += "\nConsider this specific feedback when making your classifications.\n"
        return context
    
    def clear_feedback(self):
        """Clear all feedback data"""
        self.feedback_data = self._get_empty_feedback_structure()
        self._save_feedback()
    
    def get_stats(self) -> Dict:
        """Get overall feedback statistics"""
        return self.feedback_data["feedback_stats"] 