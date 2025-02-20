import spacy
import re
from typing import List, Dict, Tuple

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_sections(text: str) -> Dict[str, str]:
    """
    Extract different sections from the resume based on headers.
    
    Args:
        text (str): Raw resume text
        
    Returns:
        dict: Dictionary with headers as keys and section content as values
    """
    # Common resume section headers
    headers = [
        'EDUCATION', 'EMPLOYMENT', 'EXPERIENCE', 'TECHNICAL EXPERIENCE',
        'SKILLS', 'PROJECTS', 'AWARDS', 'LANGUAGES', 'ADDITIONAL EXPERIENCE',
        'PERSONAL INFORMATION'
    ]
    
    sections = {}
    current_header = None
    current_content = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if line is a header
        if any(header in line.upper() for header in headers):
            if current_header:
                sections[current_header] = '\n'.join(current_content)
                current_content = []
            current_header = line.upper()
        else:
            if current_header:
                current_content.append(line)
    
    # Add the last section
    if current_header and current_content:
        sections[current_header] = '\n'.join(current_content)
        
    return sections

def clean_text(text: str) -> Tuple[str, dict]:
    """
    Clean and normalize resume text while preserving important information.
    
    Args:
        text (str): Input resume text
        
    Returns:
        tuple: (cleaned text, dict of removed items and extracted information)
    """
    metadata = {
        'personal_info': {},
        'removed_items': {
            'emails': [],
            'phone_numbers': [],
            'person_names': [],
            'locations': [],
            'dates': [],
            'urls': []
        },
        'extracted_skills': set(),
        'sections': {}
    }
    
    # Extract sections
    sections = extract_sections(text)
    metadata['sections'] = sections
    
    # Remove personal information section
    if 'PERSONAL INFORMATION' in sections:
        text = text.replace(sections['PERSONAL INFORMATION'], '')
    
    # Remove email addresses
    emails = re.findall(r'\S+@\S+\.\S+', text)
    metadata['removed_items']['emails'].extend(emails)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove phone numbers
    phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phones = re.findall(phone_pattern, text)
    metadata['removed_items']['phone_numbers'].extend(phones)
    text = re.sub(phone_pattern, '', text)
    
    # Process with spaCy
    doc = nlp(text)
    
    # Technical skills patterns
    skill_patterns = [
        r'python|java|c\+\+|javascript|sql|html|css|docker|kubernetes|aws|azure|git|react|angular|vue|node\.js',
        r'machine learning|deep learning|artificial intelligence|data science|cloud computing|devops|agile|scrum',
        r'mysql|postgresql|mongodb|redis|elasticsearch|kafka|spark|hadoop|tensorflow|pytorch|scikit-learn'
    ]
    
    cleaned_tokens = []
    for token in doc:
        # Extract entities
        if token.ent_type_:
            if token.ent_type_ == 'PERSON':
                metadata['removed_items']['person_names'].append(token.text)
                continue
            elif token.ent_type_ == 'GPE' or token.ent_type_ == 'LOC':
                metadata['removed_items']['locations'].append(token.text)
                continue
            elif token.ent_type_ == 'DATE':
                metadata['removed_items']['dates'].append(token.text)
                continue
        
        # Extract technical skills
        token_lower = token.text.lower()
        for pattern in skill_patterns:
            if re.search(pattern, token_lower):
                metadata['extracted_skills'].add(token_lower)
        
        # Skip unwanted tokens
        if (token.is_punct or token.is_space or 
            token.like_num or len(token.text) <= 1):
            continue
        
        # Normalize and add token
        cleaned_token = token.text.lower().strip()
        if cleaned_token:
            cleaned_tokens.append(cleaned_token)
    
    # Convert skills set to list for JSON serialization
    metadata['extracted_skills'] = list(metadata['extracted_skills'])
    
    return ' '.join(cleaned_tokens), metadata

def extract_entities(text: str) -> dict:
    """
    Extract named entities from text using spaCy with focus on resume-relevant entities.
    
    Args:
        text (str): Input text to process
        
    Returns:
        dict: Dictionary of entity types and their values
    """
    doc = nlp(text)
    entities = {
        'SKILL': [],
        'ORG': [],
        'DATE': [],
        'GPE': [],
        'DEGREE': [],
        'LANGUAGE': []
    }
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    return entities
