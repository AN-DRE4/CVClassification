import spacy
import re
from spacy.tokens import Doc, Span

# Load spaCy for tokenization
nlp = spacy.load("en_core_web_lg")

def create_labeled_spans(doc_text, extracted_info):
    doc = nlp(doc_text)
    doc.spans["skills"] = []
    doc.spans["companies"] = []
    doc.spans["job_titles"] = []
    doc.spans["dates"] = []
    
    # Label skills
    if "skills" in extracted_info:
        for skill in extracted_info["skills"]:
            skill_matches = find_spans(doc, skill)
            for start, end in skill_matches:
                doc.spans["skills"].append(Span(doc, start, end, label="SKILL"))
    
    # Label work experience
    if "work_experience" in extracted_info:
        for exp in extracted_info["work_experience"]:
            if "company" in exp:
                company_matches = find_spans(doc, exp["company"])
                for start, end in company_matches:
                    doc.spans["companies"].append(Span(doc, start, end, label="COMPANY"))
            
            # Similarly process job titles, dates, etc.
            # ...
    
    return doc

def find_spans(doc, text):
    """Find all occurrences of text in doc, returning token spans"""
    text = text.lower()
    matches = []
    
    # Simple exact matching (you might want more sophisticated matching)
    doc_text = doc.text.lower()
    for match in re.finditer(re.escape(text), doc_text):
        start_char = match.start()
        end_char = match.end()
        
        # Find token indices that correspond to these character positions
        start_token = None
        end_token = None
        
        for i, token in enumerate(doc):
            if token.idx <= start_char < token.idx + len(token.text):
                start_token = i
            if token.idx <= end_char <= token.idx + len(token.text) and end_token is None:
                end_token = i + 1
        
        if start_token is not None and end_token is not None:
            matches.append((start_token, end_token))
    
    return matches

# Apply to all silver data
labeled_docs = []
for _, row in silver_df.iterrows():
    doc = create_labeled_spans(row['resume_text'], row['extracted_info'])
    labeled_docs.append(doc)