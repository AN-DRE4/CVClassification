import json
import os
from typing import List, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class HumanLabelingInterface:
    def __init__(self, output_dir: str = "labeled_samples"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_samples_for_labeling(self, samples: List[Dict[str, Any]], iteration: int):
        """Save samples to be labeled by humans"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iteration_{iteration}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(samples, f, indent=2)
        
        logger.info(f"Saved {len(samples)} samples for labeling to {filepath}")
        return filepath
    
    def load_labeled_samples(self, filepath: str) -> List[Dict[str, Any]]:
        """Load samples that have been labeled by humans"""
        if not os.path.exists(filepath):
            logger.error(f"Labeled samples file not found: {filepath}")
            return []
        
        with open(filepath, 'r') as f:
            labeled_samples = json.load(f)
        
        # Validate labeled samples
        valid_samples = []
        for sample in labeled_samples:
            if self._validate_sample(sample):
                valid_samples.append(sample)
            else:
                logger.warning(f"Invalid sample format found in {filepath}")
        
        logger.info(f"Loaded {len(valid_samples)} valid labeled samples from {filepath}")
        return valid_samples
    
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate that a sample has the required structure"""
        required_fields = ['resume_id', 'resume_text', 'extracted_info']
        if not all(field in sample for field in required_fields):
            return False
        
        extracted_info = sample['extracted_info']
        if not isinstance(extracted_info, dict):
            return False
        
        # Validate work experience structure
        if 'work_experience' in extracted_info:
            for job in extracted_info['work_experience']:
                if not isinstance(job, dict):
                    return False
                required_job_fields = ['company', 'title', 'start_date', 'end_date', 
                                     'responsibilities', 'technical_skills', 'soft_skills']
                if not all(field in job for field in required_job_fields):
                    return False
        
        # Validate extra skills structure
        if 'extra_skills' in extracted_info:
            extra_skills = extracted_info['extra_skills']
            if not isinstance(extra_skills, dict):
                return False
            if not all(skill_type in extra_skills for skill_type in ['Technical skills', 'Soft skills']):
                return False
        
        return True
    
    def get_labeling_instructions(self) -> str:
        """Return instructions for human labelers"""
        return """
        Instructions for Labeling Resumes:
        
        1. For each resume, analyze the work experience and skills sections to determine:
           - Technical skills used in each job
           - Soft skills demonstrated
           - Extra skills not associated with specific jobs
        
        2. For each work experience entry:
           - Company name: The full company name
           - Job title: The exact job title
           - Start date: Month Year format (e.g., "January 2020")
           - End date: Month Year format or "Present" if current
           - Duration: Years (rounded to 0.5)
           - Responsibilities: List of key responsibilities
           - Technical skills: List of technical skills used in this role
           - Soft skills: List of soft skills demonstrated in this role
        
        3. For extra skills:
           - Technical skills: List all technical skills not associated with specific jobs
           - Soft skills: List all soft skills not associated with specific jobs
        
        4. Format:
           - All dates should be in "Month Year" format
           - All lists should be arrays of strings
           - Skills should be specific and technical where possible
           - Responsibilities should be clear and concise
        
        5. Example structure:
        {
            "resume_id": "path/to/resume.txt",
            "resume_text": "full resume text",
            "extracted_info": {
                "work_experience": [
                    {
                        "company": "Example Corp",
                        "title": "Senior Software Engineer",
                        "start_date": "January 2020",
                        "end_date": "Present",
                        "duration": 3.5,
                        "responsibilities": ["Lead development team", "Design system architecture"],
                        "technical_skills": ["Python", "Docker", "AWS"],
                        "soft_skills": ["Leadership", "Communication"]
                    }
                ],
                "extra_skills": {
                    "Technical skills": ["Git", "Linux"],
                    "Soft skills": ["Problem Solving"]
                }
            }
        }
        """ 