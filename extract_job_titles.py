import json
import re
import os
from collections import defaultdict

# Role level mapping: keys are standardized role levels, values are lists of keywords to match
ROLE_LEVEL_MAPPING = {
    "entry_level": ["entry", "junior", "jr", "intern", "trainee", "fresher", "graduate", "assistant", "clerk"],
    "mid_level": ["associate", "staff", "analyst", "consultant", "specialist"],
    "senior_level": ["senior", "sr", "lead", "principal", "tech lead", "technical lead", "expert", "team leader", "team lead"],
    "management": ["manager", "director", "vp", "chief", "head", "executive", "officer", "president"],
}

ROLE_HIERARCHY = ["management", "senior_level", "mid_level", "entry_level", "uncategorized"]

EXTRA_MAPPING = [
    "developer",
    "designer",
    "qa",
    "hr",
    "human resources",
    "analyst",
    "accountant",
    "consultant",
    "marketing",
    "sales",
    "finance",
]

def extract_role_level(title):
    """
    Extract the role level from a job title based on the role level mapping.
    If no known role level is found, return "uncategorized".
    """
    ret = ["uncategorized"]

    if not title or not isinstance(title, str):
        return ret
    
    title_lower = title.lower()
    
    # Check each role level category
    for role_level, keywords in ROLE_LEVEL_MAPPING.items():
        for keyword in keywords:
            # Match keyword as a whole word
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, title_lower):
                ret.append(role_level)
    
    # If no predefined level is found
    return ret

def process_resumes(input_file, output_file):
    """
    Process resume data, extract job titles and categorize them by role level.
    """
    print(f"Processing resumes from {input_file}...")
    
    # Check if file exists and is not empty
    if not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
        print(f"Error: Input file {input_file} does not exist or is empty")
        return False
    
    try:
        # Load the JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            resumes = json.load(f)
        
        print(f"Loaded {len(resumes)} resumes")
        
        # Dictionary to store titles by role level
        titles_by_level = defaultdict(set)
        
        # Process each resume
        for i, resume in enumerate(resumes):
            if i % 1000 == 0:
                print(f"Processing resume {i}/{len(resumes)}")
            
            # Check if resume has the expected structure
            if not isinstance(resume, dict) or "extracted_info" not in resume:
                continue
            
            extracted_info = resume.get("extracted_info", {})
            work_experiences = extracted_info.get("work_experience", [])
            
            # Extract titles from work experience
            for job in work_experiences:
                if not isinstance(job, dict):
                    continue
                
                title = job.get("title")
                if title and title != "N/A":
                    # Extract role level
                    role_levels = extract_role_level(title)
                    role_level = role_levels[0]
                    if len(role_levels) == 1: # only caught has 'uncategorized'
                        for item in EXTRA_MAPPING:
                            if item in title.lower():
                                role_level = 'mid_level'
                                print(f"{title} -> {role_levels} -> {role_level}")
                                break
                    else:
                        # Take only the highest role level according to hierarchy
                        for level in ROLE_HIERARCHY:
                            if level in role_levels:
                                role_level = level
                                break
                    titles_by_level[role_level].add(title)
        
        # Convert sets to lists for JSON serialization
        result = {level: list(titles) for level, titles in titles_by_level.items()}
        
        # Save results to a new JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(f"Role level categorization completed. Results saved to {output_file}")
        
        # Print summary
        print("\nSummary of categorization:")
        for level, titles in result.items():
            print(f"{level}: {len(titles)} unique titles")
        
        return True
    
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON data in {input_file}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    
"""def check_job_titles(title, categories):
    title_new = title.lower()
    if title_new in categories['management'] or title_new in categories['senior_level'] or title_new in categories['mid_level'] or title_new in categories['entry_level']:
        return True
    return False
"""

def interactive_categorization(input_file, output_file):
    """
    Allow human intervention for categorizing job titles.
    """
    try:
        # Check if output file already exists
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                categories = json.load(f)
        else:
            categories = {}
        
        # Load the JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            resumes = json.load(f)
        
        # Set to keep track of already properly categorized titles
        already_categorized_titles = set()
        
        # Only skip titles that have been categorized with a proper category (not 'uncategorized')
        for category, titles in categories.items():
            if category != "uncategorized":
                already_categorized_titles.update(titles)
        
        # Dictionary to store titles to be categorized in this run
        titles_to_categorize = set()
        
        # Get all uncategorized titles from the existing categories
        uncategorized_titles = set(categories.get("uncategorized", []))

        """titles_to_remove = []
        print(f"Uncategorized titles: {len(uncategorized_titles)}")
        for title in uncategorized_titles:
            print(title)
            if check_job_titles(title, categories):
                titles_to_remove.append(title)
        
        print(f"There are {len(titles_to_remove)} titles that are already categorized with different casing and will be removed")
        # Output titles that will be removed to a JSON file
        with open('removed_titles.json', 'w', encoding='utf-8') as f:
            json.dump({'removed_titles': titles_to_remove}, f, indent=2)
        
        num = input("Press 1 to remove the titles and anything else to keep them").strip()
        if num == '1':
            for title in titles_to_remove:
                uncategorized_titles.remove(title)
                categories['uncategorized'].remove(title)

        print(f"Uncategorized titles after checking: {len(uncategorized_titles)}")"""
        # Process each resume
        for resume in resumes:
            if "extracted_info" not in resume:
                continue
            
            extracted_info = resume.get("extracted_info", {})
            work_experiences = extracted_info.get("work_experience", [])
            
            # Extract titles from work experience
            for job in work_experiences:
                if not isinstance(job, dict):
                    continue
                
                title = job.get("title")
                # Only process titles that aren't already properly categorized
                if title and title != "N/A" and title not in already_categorized_titles:
                    titles_to_categorize.add(title)
        
        # If there are titles to categorize, process them
        if titles_to_categorize:
            print(f"Found {len(titles_to_categorize)} job titles to categorize.")
            
            # If there are uncategorized titles in the existing data, report them
            uncategorized_to_process = uncategorized_titles.intersection(titles_to_categorize)
            new_titles_to_process = titles_to_categorize - uncategorized_titles
            
            if uncategorized_to_process:
                print(f"- {len(uncategorized_to_process)} were previously marked as 'uncategorized' and will be re-categorized")
            if new_titles_to_process:
                print(f"- {len(new_titles_to_process)} are new titles that haven't been categorized before")
            
            # Display available role levels
            print("\nAvailable standard role levels:")
            for role_level, keywords in ROLE_LEVEL_MAPPING.items():
                print(f"- {role_level}: {', '.join(keywords)}")
            print("- uncategorized")
            
            # Define category keys for menu selection
            category_keys = ["entry_level", "mid_level", "senior_level", "management", "uncategorized"]
            
            # Process all titles that need categorization
            for i, title in enumerate(sorted(titles_to_categorize), 1):
                suggested_level = extract_role_level(title)
                
                print(f"\nTitle: {title}")
                print(f"Suggested level: {suggested_level}")
                
                # If this title was previously uncategorized, mention that
                if title in uncategorized_titles:
                    print("Note: This title was previously marked as 'uncategorized'")
                
                user_input = input(f"""
Enter role level for this title (press Enter to accept '{suggested_level}'):
1 - Entry Level
2 - Mid Level
3 - Senior Level
4 - Management
5 - Uncategorized
6 - Exit
""").strip()
                
                # Process user input
                chosen_category = ""
                
                # Use suggested level if the user just presses Enter
                if not user_input:
                    chosen_category = suggested_level
                # If user entered a number 1-5, map it to a category
                elif user_input.isdigit() and 1 <= int(user_input) <= 6:
                    if int(user_input) == 6:
                        print(f"Exiting interactive categorization with {len(titles_to_categorize) - i + 1} job titles remaining...")
                        break
                    chosen_category = category_keys[int(user_input) - 1]
                # Otherwise, use the input as a custom category
                else:
                    chosen_category = user_input.lower()
                
                # Remove the title from any existing category (including "uncategorized")
                for category, title_list in categories.items():
                    if title in title_list:
                        title_list.remove(title)
                
                # Add title to the appropriate category
                if chosen_category not in categories:
                    categories[chosen_category] = []
                
                categories[chosen_category].append(title)
        else:
            print("No job titles found to categorize.")
        
        # Clean up empty categories
        categories = {category: titles for category, titles in categories.items() if titles}
        
        # Save updated categories to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(categories, f, indent=2)
        
        print(f"Updated role level categorization saved to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error during interactive categorization: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and categorize job titles from resume data")
    parser.add_argument("--input", default="silver_labeled_resumes.json", help="Input JSON file with resume data")
    parser.add_argument("--output", default="job_titles_by_role_level.json", help="Output JSON file for categorized titles")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode for human categorization")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_categorization(args.input, args.output)
    else:
        process_resumes(args.input, args.output) 