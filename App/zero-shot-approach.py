import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

year = datetime.now().year
month = datetime.now().strftime("%B")

print(month, year)

def extract_resume_info(resume_text):
    prompt = """
    Today's date is {month} {year}.
    Extract the following structured information from this resume:
        
    1. Education: Degrees, institutions, and graduation dates
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
        print(f"Error processing resume: {e}")
        return None


resumes = []
for root, dirs, files in os.walk('CVs'):
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                resume_text = f.read()
                resumes.append({
                    'id': os.path.join(root, file),  # Use path as ID
                    'text': resume_text
                })

for resume in resumes[:5]:
    print(resume['id'])

print(len(resumes))

# Process a subset of resumes
subset_size = 5
resumes_subset = resumes[:subset_size]  # First 500 resumes
processed_data = []

for resume in tqdm(resumes_subset):
    extracted_info = extract_resume_info(resume['text'])
    if extracted_info:
        processed_data.append({
            'resume_id': resume['id'],
            'resume_text': resume['text'],
            'extracted_info': extracted_info
        })

# Save as silver data
silver_df = pd.DataFrame(processed_data)
silver_df.to_json('silver_labeled_resumes2.json', orient='records')