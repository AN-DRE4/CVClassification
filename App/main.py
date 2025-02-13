from resume_parsing import parse_resume_pdf
from text_cleaning import clean_text, extract_entities
import streamlit as st
import tempfile
import os

def streamlit_display():
    st.title("Resume Parser")
    
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a PDF resume", type=['pdf'])
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process the resume
        content, elements = parse_resume_pdf(tmp_path, "outputs/out3.txt")

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
        
        # Clean up the temporary file
        os.unlink(tmp_path)


def main():
    streamlit_display()
if __name__ == "__main__":
    main()