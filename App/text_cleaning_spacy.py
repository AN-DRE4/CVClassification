import spacy
import re
from typing import List

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# a lot of valuable information is lost here, but it's a good start
# TODO: add more sophisticated text cleaning, try to keep as much valuable information as possible
# i.e. technical skills, education, experience, tools, etc.
def clean_text(text: str) -> str:
    """
    Clean and normalize text using spaCy. Removes headers, personal names, emails and normalizes text.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned and normalized text with headers, names and emails removed, and a dict of removed items
    """
    removed_items = {
        'emails': [],
        'person_names': [],
        'headers': [],
        'punctuation': []
    }
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove email addresses
    emails = re.findall(r'\S+@\S+\.\S+', text)
    removed_items['emails'].extend(emails)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Process with spaCy
    doc = nlp(text)
    
    # Clean tokens
    cleaned_tokens = []
    for token in doc:
        # Track punctuation
        if token.is_punct:
            removed_items['punctuation'].append(token.text)
            continue
            
        # Track person names    
        if token.ent_type_ == 'PERSON':
            removed_items['person_names'].append(token.text)
            continue
            
        # Track headers
        header_words = ['education', 'experience', 'skills', 
                       'employment', 'awards', 'languages',
                       'technical', 'additional']
        if (token.text.lower() in header_words or 
            (token.text.isupper() and len(token.text) > 2)):
            removed_items['headers'].append(token.text)
            continue
            
        if token.is_space:
            continue
        
        # Normalize token (lowercase, remove extra spaces)
        cleaned_token = token.text.lower().strip()
        if cleaned_token:
            cleaned_tokens.append(cleaned_token)
    
    return ' '.join(cleaned_tokens), removed_items

def extract_entities(text: str) -> dict:
    """
    Extract named entities from text using spaCy.
    
    Args:
        text (str): Input text to process
        
    Returns:
        dict: Dictionary of entity types and their values
    """
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    
    return entities
