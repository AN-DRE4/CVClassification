import json
from icecream import ic
from cv_agents.expertise.agent import ExpertiseAgent
from cv_agents.role.agent import RoleLevelAgent
from cv_agents.org_unit.agent import OrgUnitAgent
from cv_agents.utils.data_extractor import extract_cv_sections

def test_individual_agents():
    """Test each agent independently"""
    # Load sample CV
    # path = "silver_labeled_resumes.json"
    path = "labeled_resume.json"
    with open(path, "r") as f:
        cv_data = json.load(f)[0]  # Use first CV for testing
    
    # Extract sections
    cv_sections = extract_cv_sections(cv_data)
    
    # Test expertise agent
    print("Testing Expertise Agent...")
    expertise_agent = ExpertiseAgent()
    expertise_results = expertise_agent.process(cv_sections)
    print(json.dumps(expertise_results, indent=2))
    
    # Test role level agent
    print("\nTesting Role Level Agent...")
    role_agent = RoleLevelAgent()
    role_input = {**cv_sections, "expertise_results": expertise_results}
    role_results = role_agent.process(role_input)
    print(json.dumps(role_results, indent=2))
    
    # Test org unit agent
    print("\nTesting Org Unit Agent...")
    org_agent = OrgUnitAgent()
    org_input = {**cv_sections, "expertise_results": expertise_results, "role_results": role_results}
    org_results = org_agent.process(org_input)
    print(json.dumps(org_results, indent=2))

if __name__ == "__main__":
    test_individual_agents()
