from ..expertise.agent import ExpertiseAgent
from ..role.agent import RoleLevelAgent  # To be implemented
from ..org_unit.agent import OrgUnitAgent  # To be implemented
from ..utils.data_extractor import extract_cv_sections

class CVClassificationChain:
    def __init__(self):
        self.expertise_agent = ExpertiseAgent()
        self.role_level_agent = RoleLevelAgent()
        self.org_unit_agent = OrgUnitAgent()
    
    def process_cv(self, cv_data):
        """Process a CV through the entire agent chain"""
        # Extract CV sections
        cv_sections = extract_cv_sections(cv_data)
        
        # Step 1: Determine expertise areas
        expertise_results = self.expertise_agent.process(cv_sections)
        
        # Step 2: Determine role level for each expertise
        # Pass both CV and expertise results to role agent
        role_input = {**cv_sections, "expertise_results": expertise_results}
        role_results = self.role_level_agent.process(role_input)
        
        # Step 3: Determine organizational unit
        # Pass CV, expertise, and role results to org unit agent
        org_input = {**cv_sections, "expertise_results": expertise_results, "role_results": role_results}
        org_results = self.org_unit_agent.process(org_input)
        
        # Combine all results
        return {
            "resume_id": cv_data.get("resume_id", ""),
            "expertise": expertise_results,
            "role_levels": role_results,
            "org_unit": org_results
        }
