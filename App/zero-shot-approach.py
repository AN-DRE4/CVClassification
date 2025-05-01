import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os
from dotenv import load_dotenv
from datetime import datetime
import time
import argparse

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

year = datetime.now().year
month = datetime.now().strftime("%B")

print(f"Processing resumes in {month} {year}")

def extract_resume_info(resume_text):
    prompt = """
    Today's date is {month} {year}.
    Extract the following structured information from this resume:
        
    1. Education: Degrees, institutions, and graduation dates. All graduation dates should be in the format 'Year' only and nothing else.
    2. Work Experience: For each position include:
       - Company name
       - Job title
       - Start date (in format 'Month Year' or 'N/A' if not provided)
       - End date (in format 'Month Year' or 'N/A' if not provided, if currently working, use '{month} {year}')
       - Duration (in years, rounded to the nearest 0.5 year, if not provided, use 'N/A', if end date is 'Present', use today's date as end date and calculate duration based both dates)
       - Key responsibilities
       - Technical skills 
       - Soft skills
    
    For the skills section in the work experience section, list all technical skills, programming languages, tools, and soft skills. 
    For each skill atribute those to the following categories and only those categories.
    If a skill does not fit into any of the categories, include it in the one that is most relevant:
       - Technical skills (e.g. programming languages, tools, etc.)
       - Soft skills (e.g. communication, teamwork, etc.) 
    Below is an example of how to format the skills output:
    {
        "Technical skills": [
            "Python", 
            "Java", 
            "C++", 
            "C#", 
            "SQL",
            ...
        ],
        "Soft skills": [
            "Communication", 
            "Teamwork", 
            "Leadership", 
            "Problem Solving",
            ...
        ]
    }
    3. ExtraSkills: List extra skills that are not related to work experience but that the candidate has.
    Here only list skills that are listed in the resume but don't correlate with any work experience the candidate has.
    List these in the same format as the skills in the work experience section.
    
    Format the output as a JSON object with these exact keys: "education", "work_experience", "extra_skills".
    Make sure each work experience entry has "company", "title", "start_date", "end_date", "responsibilities", "technical_skills" and "soft_skills" fields.
    """
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": """You are an expert resume analyst. 
                     You extract structured information from resumes accurately and in a structured way.
                     You are given a resume in text format and you need to extract the information in a structured way following the prompt below. 
                     """},
                    {"role": "user", "content": prompt + "\n\nRESUME:\n" + resume_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # Keep it deterministic
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Error processing resume after {max_retries} attempts: {e}")
                return None
            
def process_folder(folder_path):
    # Load existing labeled data if it exists
    existing_data = []
    if os.path.exists('silver_labeled_resumes.json'):
        with open('silver_labeled_resumes.json', 'r') as f:
            existing_data = json.load(f)
        print(f"Loaded {len(existing_data)} existing labeled resumes")

    # Get list of already processed resume IDs
    processed_ids = {item['resume_id'] for item in existing_data}

    # Load all resumes
    resumes = []
    for root, dirs, files in os.walk('folder_path'):
        for file in files:
            if file.endswith('.txt'):
                resume_id = os.path.join(root, file)
                if resume_id not in processed_ids:  # Only add unprocessed resumes
                    with open(resume_id, 'r', encoding='utf-8') as f:
                        resume_text = f.read()
                        resumes.append({
                            'id': resume_id,
                            'text': resume_text
                        })

    print(f"Found {len(resumes)} unprocessed resumes")

    # Process all remaining resumes
    processed_data = existing_data  # Start with existing data


    print(f"Processing {len(resumes)} remaining resumes...")
    for resume in tqdm(resumes):
        extracted_info = extract_resume_info(resume['text'])
        if extracted_info:
            processed_data.append({
                'resume_id': resume['id'],
                'resume_text': resume['text'],
                'extracted_info': extracted_info
            })

    # Save as silver data
    silver_df = pd.DataFrame(processed_data)
    silver_df.to_json('silver_labeled_resumes.json', orient='records')
    print(f"Saved {len(processed_data)} total resumes to silver_labeled_resumes.json")

def process_cv(cv_path):
    with open(cv_path, 'r', encoding='utf-8') as f:
        resume_text = f.read()
    extracted_info = extract_resume_info(resume_text)
    processed_data = []
    processed_data.append({
        'resume_id': cv_path,
        'resume_text': resume_text,
        'extracted_info': extracted_info
    })
    silver_df = pd.DataFrame(processed_data)
    silver_df.to_json('labeled_resume.json', orient='records')
    print(f"Saved {len(processed_data)} total resumes to labeled_resume.json")

def main():
    parser = argparse.ArgumentParser(description='Process resumes')
    parser.add_argument('--folder', '-f', type=str, help='Path to the folder containing the resumes')
    parser.add_argument('--cv', type=str, help='Path to the CV to process')
    # parser.add_argument('--help', action='store_true', help='Show this help message and exit')
    args = parser.parse_args()

    if args.folder:
        process_folder(args.folder)
    if args.cv:
        process_cv(args.cv)


if __name__ == '__main__':
    main()
