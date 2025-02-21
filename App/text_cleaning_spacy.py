import spacy
import re
from icecream import ic
from typing import List, Dict, Tuple    
from display_spacy import display_spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_sections(text: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract different sections from the resume based on formatting rules:
    - All uppercase without indentation = Header
    - Not uppercase without indentation = Additional information
    - Any indentation = Content
    
    Args:
        text (str): Raw resume text
        
    Returns:
        dict: Dictionary with headers as keys and nested dictionaries containing
             additional_info and content
    """
    sections = {}
    current_header = None
    current_additional_info = []
    current_content = []
    
    for line in text.split('\n'):
        original_line = line
        line = line.strip()
        
        if not line:  # Skip empty lines
            continue
        
        # Check indentation (count leading spaces)
        indentation = len(original_line) - len(original_line.lstrip())
        
        # Check if line is a header (all uppercase, no indentation)
        if line.isupper() and indentation == 0:
            # Save previous section if it exists
            if current_header:
                sections[current_header] = {
                    'additional_info': current_additional_info,
                    'content': current_content
                }
            # Start new section
            current_header = line
            current_additional_info = []
            current_content = []
        
        # If line has indentation, it's content
        elif indentation > 0:
            current_content.append(line)
        
        # If line has no indentation but isn't all uppercase, it's additional info
        elif current_header:
            current_additional_info.append(line)
    
    # Add the last section
    if current_header:
        sections[current_header] = {
            'additional_info': current_additional_info,
            'content': current_content
        }
    
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
    
    # Extract sections with the new structure
    sections = extract_sections(text)
    metadata['sections'] = sections
    
    # Remove personal information section
    if 'PERSONAL INFORMATION' in sections:
        personal_info = sections['PERSONAL INFORMATION']
        # Remove both additional info and content from personal information
        for info in personal_info['additional_info']:
            text = text.replace(info, '')
        for content in personal_info['content']:
            text = text.replace(content, '')
    
    # Prepare text for cleaning by combining relevant parts
    processed_text = []
    for header, section_data in sections.items():
        if header != 'PERSONAL INFORMATION':
            # Add header
            processed_text.append(header)
            # Add additional info
            processed_text.extend(section_data['additional_info'])
            # Add content
            processed_text.extend(section_data['content'])
    
    text = ' '.join(processed_text)
    
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

    display_spacy(doc)
    
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
