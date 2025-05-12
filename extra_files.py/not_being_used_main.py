from App.not_being_used_resume_parsing import parse_resume_pdf
from App.not_being_used_text_cleaning_spacy import clean_text, extract_entities
from App.not_being_used_text_classification_transformers import classify_text
import streamlit as st
import tempfile
import os

with open("outputs/outDeepseek.txt", "r", encoding="utf-8") as file:
    temporary_test_file = file.read()

def parse_resume_pdf_spacy(tmp_path: str):# Process the resume
    # content, elements = parse_resume_pdf(tmp_path, "outputs/out3.txt")
    content = temporary_test_file
    elements = temporary_test_file

    # Clean the text
    cleaned_text, removed_items = clean_text(content)

    # Extract entities
    entities = extract_entities(cleaned_text)
    
    # Display the parsed content
    st.write("Parsed Resume Content:")
    st.write(content)

    # Display the elements
    st.write("Elements:")
    st.write(elements)

    # Display the cleaned text
    st.write("Cleaned Text:")
    st.write(cleaned_text)

    # Display the removed items
    st.write("Removed Items:")
    st.write(removed_items)

    # Display the entities
    st.write("Entities:")
    st.write(entities)

def parse_resume_pdf_transformers(tmp_path: str):
    content = temporary_test_file
    elements = temporary_test_file

    # Clean the text and get metadata
    cleaned_text, metadata = clean_text(content)

    # Use the cleaned text for classification
    result = classify_text(cleaned_text)

    # Display results
    st.write("Classification Results:")
    for item in result:
        st.write(f"{item}: {result[item]}")
    
    st.write("\nExtracted Resume Structure:")
    for header, section_data in metadata['sections'].items():
        st.write(f"\n{header}")
        if section_data['additional_info']:
            st.write("Additional Information:")
            for info in section_data['additional_info']:
                st.write(f"  - {info}")
        if section_data['content']:
            st.write("Content:")
            for content in section_data['content']:
                st.write(f"    {content}")
    
    st.write("\nExtracted Skills:", metadata['extracted_skills'])
    st.write("Removed Personal Information:", metadata['removed_items'])


def streamlit_display():
    st.title("Resume Parser")
    
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a PDF resume", type=['pdf'])
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        #parse_resume_pdf_spacy(tmp_path)
        
        parse_resume_pdf_transformers(tmp_path)

        # Clean up the temporary file
        os.unlink(tmp_path)


def main():
    streamlit_display()
if __name__ == "__main__":
    main()