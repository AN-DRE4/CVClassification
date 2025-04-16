import requests
import json
from typing import List, Dict, Optional
import logging

class ESCOIntegration:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize ESCO API integration
        
        Args:
            api_key: Optional API key for ESCO API. If not provided, will use public API
        """
        self.base_url = "https://ec.europa.eu/esco/api"
        self.api_key = api_key
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        # Cache for storing API responses
        self.skill_cache = {}
        self.occupation_cache = {}
        
    def search_skills(self, query: str, language: str = "en") -> List[Dict]:
        """
        Search for skills in ESCO taxonomy
        
        Args:
            query: Search term
            language: Language code (default: "en")
            
        Returns:
            List of matching skills with their metadata
        """
        if query in self.skill_cache:
            return self.skill_cache[query]
            
        endpoint = f"{self.base_url}/search"
        params = {
            "text": query,
            "language": language,
            "type": "skill"
        }
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            results = response.json()["_embedded"]["results"]
            self.skill_cache[query] = results
            return results
        except Exception as e:
            logging.error(f"Error searching ESCO skills: {str(e)}")
            return []
            
    def search_occupations(self, query: str, language: str = "en") -> List[Dict]:
        """
        Search for occupations in ESCO taxonomy
        
        Args:
            query: Search term
            language: Language code (default: "en")
            
        Returns:
            List of matching occupations with their metadata
        """
        if query in self.occupation_cache:
            return self.occupation_cache[query]
            
        endpoint = f"{self.base_url}/resource/occupation/search"
        params = {
            "text": query,
            "language": language,
            "type": "occupation"
        }
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            results = response.json()["_embedded"]["results"]
            self.occupation_cache[query] = results
            return results
        except Exception as e:
            logging.error(f"Error searching ESCO occupations: {str(e)}")
            return []
            
    def get_skill_details(self, skill_uri: str, language: str = "en") -> Dict:
        """
        Get detailed information about a specific skill
        
        Args:
            skill_uri: ESCO skill URI
            language: Language code (default: "en")
            
        Returns:
            Detailed skill information
        """
        endpoint = f"{self.base_url}/resource/skill/{skill_uri}"
        params = {"language": language}
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error getting skill details: {str(e)}")
            return {}
            
    def get_occupation_details(self, occupation_uri: str, language: str = "en") -> Dict:
        """
        Get detailed information about a specific occupation
        
        Args:
            occupation_uri: ESCO occupation URI
            language: Language code (default: "en")
            
        Returns:
            Detailed occupation information
        """
        endpoint = f"{self.base_url}/resource/occupation/{occupation_uri}"
        params = {"language": language}
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error getting occupation details: {str(e)}")
            return {}
            
    def map_skills_to_esco(self, skills: List[str], threshold: float = 0.7) -> Dict[str, List[Dict]]:
        """
        Map a list of skills to ESCO taxonomy
        
        Args:
            skills: List of skills to map
            threshold: Similarity threshold for matching (default: 0.7)
            
        Returns:
            Dictionary mapping input skills to ESCO matches
        """
        mapped_skills = {}
        
        for skill in skills:
            matches = self.search_skills(skill)
            if matches:
                # Filter matches by similarity score
                filtered_matches = [
                    match for match in matches 
                    if match.get("relevanceScore", 0) >= threshold
                ]
                if filtered_matches:
                    mapped_skills[skill] = filtered_matches
                    
        return mapped_skills
        
    def map_job_title_to_esco(self, job_title: str, threshold: float = 0.7) -> List[Dict]:
        """
        Map a job title to ESCO occupations
        
        Args:
            job_title: Job title to map
            threshold: Similarity threshold for matching (default: 0.7)
            
        Returns:
            List of matching ESCO occupations
        """
        matches = self.search_occupations(job_title)
        if matches:
            return [
                match for match in matches 
                if match.get("relevanceScore", 0) >= threshold
            ]
        return []
        
    def get_related_skills(self, occupation_uri: str, language: str = "en") -> List[Dict]:
        """
        Get skills related to a specific occupation
        
        Args:
            occupation_uri: ESCO occupation URI
            language: Language code (default: "en")
            
        Returns:
            List of related skills
        """
        occupation_details = self.get_occupation_details(occupation_uri, language)
        if occupation_details:
            return occupation_details.get("hasEssentialSkill", []) + \
                   occupation_details.get("hasOptionalSkill", [])
        return [] 
    
def main():
    esco = ESCOIntegration()
    print(esco.map_skills_to_esco(["python"]))
    
if __name__ == "__main__":
    main()
