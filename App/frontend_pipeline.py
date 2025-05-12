import streamlit as st
import json
import os
from datetime import datetime
import sys
import importlib.util

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project components - handle file with hyphen in name
zero_shot_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zero-shot-approach.py")
spec = importlib.util.spec_from_file_location("zero_shot_approach", zero_shot_module_path)
zero_shot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zero_shot_module)
extract_resume_info = zero_shot_module.extract_resume_info

from process_cvs import CVProcessor

# Set page configuration
st.set_page_config(
    page_title="CV Classification Pipeline",
    page_icon="📄",
    layout="wide"
)

# Initialize orchestrator
@st.cache_resource
def get_orchestrator(input_file: str, output_dir: str = "results"):
    return CVProcessor(input_file, output_dir)

def process_cv_text(cv_text, file_name):
    """Process a CV text through the entire pipeline"""
    # Step 1: Initial processing with zero-shot approach
    st.write("### Step 1: Initial CV information extraction")
    with st.spinner("Extracting CV information..."):
        extracted_info = extract_resume_info(cv_text)
    
    if not extracted_info:
        st.error("Failed to extract information from the CV")
        return None
    
    # Display extracted info
    st.json(extracted_info)
    
    # Step 2: Create a processed CV object
    cv_data = {
        'resume_id': f"upload_{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'resume_text': cv_text,
        'extracted_info': extracted_info
    }

    cv_data = [cv_data]

    # save the processed CV data to a file
    with open("processed_cv_data.json", "w") as f:
        json.dump(cv_data, f)
    
    # Step 3: Process with agents/vector search
    st.write("### Step 2: Advanced CV Classification")
    with st.spinner("Classifying CV with agents..."):
        orchestrator = get_orchestrator(input_file="processed_cv_data.json", output_dir="agents_results")
        try:
            results = orchestrator.process_cvs(batch_size=1, save_interval=1, max_cvs=1)
            return results
        except Exception as e:
            st.error(f"Error during CV classification: {str(e)}")
            return None

def display_results(results):
    """Display the classification results in a formatted way"""
    st.write("## Classification Results")
    
    # Source of classification
    st.write(f"**Source:** {results.get('source', 'Direct analysis')}")

    st.write("### Percentage indicates the confidence of the classification")
    
    # Create three columns for the three main categories
    col1, col2, col3 = st.columns(3)
    
    # Expertise Areas
    with col1:
        st.write("### Expertise Areas")
        expertise_list = results.get("expertise", {}).get("expertise", [])
        if expertise_list:
            # Sort by confidence
            sorted_expertise = sorted(expertise_list, key=lambda x: x.get("confidence", 0), reverse=True)
            for exp in sorted_expertise:
                conf_pct = int(exp.get("confidence", 0) * 100)
                st.write(f"{exp.get('category')} ({conf_pct}%)")
                st.progress(exp.get("confidence", 0))
        else:
            st.write("No expertise areas identified")
    
    # Role Levels
    with col2:
        st.write("### Role Levels")
        role_levels = results.get("role_levels", {}).get("role_levels", [])
        if role_levels:
            # Sort by confidence
            sorted_roles = sorted(role_levels, key=lambda x: x.get("confidence", 0), reverse=True)
            for role in sorted_roles:
                conf_pct = int(role.get("confidence", 0) * 100)
                st.write(f"**{role.get('expertise')}:** {role.get('level')} ({conf_pct}%)")
                st.progress(role.get("confidence", 0))
        else:
            st.write("No role levels identified")
    
    # Organizational Units
    with col3:
        st.write("### Organizational Units")
        org_units = results.get("org_unit", {}).get("org_units", [])
        if org_units:
            # Sort by confidence
            sorted_units = sorted(org_units, key=lambda x: x.get("confidence", 0), reverse=True)
            for unit in sorted_units:
                conf_pct = int(unit.get("confidence", 0) * 100)
                st.write(f"{unit.get('unit')} ({conf_pct}%)")
                st.progress(unit.get("confidence", 0))
        else:
            st.write("No organizational units identified")

def main():
    st.title("CV Classification Pipeline")
    st.write("""
    Upload a CV to classify it using our AI-powered pipeline. The system will:
    1. Extract key information from your CV
    2. Classify your expertise areas, role levels, and organizational unit fit
    """)
    
    # File uploader for CV text
    uploaded_file = st.file_uploader("Upload your CV (text format)", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        file_name = uploaded_file.name
    else:
        file_name = None
    
    # Text area for direct input
    st.write("#### Or paste your CV text below")
    cv_text = st.text_area("CV Text", height=200)
    
    # Process button
    process_clicked = st.button("Process CV")
    
    if process_clicked:
        # Get CV text from either file or text area
        if uploaded_file is not None:
            cv_text = uploaded_file.getvalue().decode("utf-8")
        
        if not cv_text:
            st.warning("Please upload a file or paste CV text")
            return
        
        # Process the CV
        with st.expander("View raw CV text", expanded=False):
            st.text(cv_text)
        
        results = process_cv_text(cv_text, file_name)[-1]
        
        if results:
            display_results(results)
            
            # Option to download results
            st.download_button(
                label="Download Results as JSON",
                data=json.dumps(results, indent=2),
                file_name=f"cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
