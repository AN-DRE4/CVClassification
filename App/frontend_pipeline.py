import streamlit as st
import json
import os
from datetime import datetime
import sys
import importlib.util
from typing import Dict, Any, Optional, List, Tuple

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
def get_orchestrator(input_file: str, output_dir: str = "results", custom_config=None, config_files=None, interpreter_configs=None):
    return CVProcessor(
        input_file=input_file, 
        output_dir=output_dir, 
        custom_config=custom_config, 
        config_files=config_files,
        interpreter_configs=interpreter_configs
    )

def process_cv_text(cv_text, file_name, custom_config=None, config_files=None, interpreter_configs=None):
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
        orchestrator = get_orchestrator(
            input_file="processed_cv_data.json", 
            output_dir="agents_results",
            custom_config=custom_config,
            config_files=config_files,
            interpreter_configs=interpreter_configs
        )
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

def create_custom_expertise_config():
    """Create a custom expertise configuration based on user input"""
    st.write("### Define Custom Expertise Categories")
    st.write("Enter the expertise categories you want to use for classification, one per line:")
    
    expertise_text = st.text_area("Expertise Categories", 
                                value="software_development\ndata_engineering\ndata_science\ndevops\ncybersecurity\nmarketing\nfinance\nmanagement",
                                height=200)
    
    if expertise_text:
        # Parse the text into a list of categories
        categories = [cat.strip() for cat in expertise_text.split('\n') if cat.strip()]
        if categories:
            return {"expertise_categories": categories}
    
    return None

def create_custom_role_levels_config():
    """Create a custom role levels configuration based on user input"""
    st.write("### Define Custom Role Levels")
    st.write("Enter the role levels you want to use for classification, in the format: name:description (one per line)")
    
    role_levels_text = st.text_area("Role Levels", 
                                  value="entry_level:Junior positions, 0-2 years experience\nmid_level:Regular positions, 2-5 years experience\nsenior_level:Senior positions, 5+ years experience\nmanagement:Management positions at any level",
                                  height=200)
    
    if role_levels_text:
        role_levels = []
        for line in role_levels_text.split('\n'):
            if ':' in line:
                name, description = line.split(':', 1)
                role_levels.append({
                    "name": name.strip(),
                    "description": description.strip()
                })
        
        if role_levels:
            return {"role_levels": role_levels}
    
    return None

def create_custom_org_units_config():
    """Create a custom org units configuration based on user input"""
    st.write("### Define Custom Organizational Units")
    st.write("Enter the organizational units you want to use for classification, in the format: name:description (one per line)")
    
    org_units_text = st.text_area("Organizational Units", 
                                value="engineering:Software development, DevOps, infrastructure\ndata:Data engineering, data science, analytics\nmarketing_sales:Marketing, sales, communications\nfinance_accounting:Finance, accounting, auditing\noperations:Project management, operations, logistics\ncustomer_service:Customer support, account management\nhr:Human resources, recruitment, training",
                                height=200)
    
    if org_units_text:
        org_units = []
        for line in org_units_text.split('\n'):
            if ':' in line:
                name, description = line.split(':', 1)
                org_units.append({
                    "name": name.strip(),
                    "description": description.strip()
                })
        
        if org_units:
            return {"org_units": org_units}
    
    return None

def create_interpreter_config():
    """Create configurations using the interpreter agent"""
    interpreter_configs = []
    
    st.write("### Expertise Categories File Interpretation")
    with st.expander("Configure Expertise Categories using a File", expanded=False):
        expertise_file = st.file_uploader("Upload file with expertise categories", type=["txt", "csv", "json", "xlsx", "md"])
        if expertise_file:
            expertise_description = st.text_area(
                "Describe how to interpret this file for expertise categories",
                "This file contains expertise categories for CV classification. Each line represents a distinct expertise area."
            )
            
            if expertise_description:
                # Save the file temporarily
                expertise_file_path = f"temp_expertise_{expertise_file.name}"
                with open(expertise_file_path, "wb") as f:
                    f.write(expertise_file.getbuffer())
                
                # Add to interpreter configs
                interpreter_configs.append((expertise_file_path, expertise_description, "expertise"))
                st.success(f"Added expertise configuration from file: {expertise_file.name}")

    st.write("### Role Levels File Interpretation")
    with st.expander("Configure Role Levels using a File", expanded=False):
        role_file = st.file_uploader("Upload file with role levels", type=["txt", "csv", "json", "xlsx", "md"])
        if role_file:
            role_description = st.text_area(
                "Describe how to interpret this file for role levels",
                "This file contains role levels for CV classification. Each section describes a role level with its requirements and responsibilities."
            )
            
            if role_description:
                # Save the file temporarily
                role_file_path = f"temp_role_{role_file.name}"
                with open(role_file_path, "wb") as f:
                    f.write(role_file.getbuffer())
                
                # Add to interpreter configs
                interpreter_configs.append((role_file_path, role_description, "role_levels"))
                st.success(f"Added role levels configuration from file: {role_file.name}")

    st.write("### Organizational Units File Interpretation")
    with st.expander("Configure Organizational Units using a File", expanded=False):
        org_file = st.file_uploader("Upload file with organizational units", type=["txt", "csv", "json", "xlsx", "md"])
        if org_file:
            org_description = st.text_area(
                "Describe how to interpret this file for organizational units",
                "This file contains organizational units for CV classification. Each section describes a unit with the skills needed for that unit."
            )
            
            if org_description:
                # Save the file temporarily
                org_file_path = f"temp_org_{org_file.name}"
                with open(org_file_path, "wb") as f:
                    f.write(org_file.getbuffer())
                
                # Add to interpreter configs
                interpreter_configs.append((org_file_path, org_description, "org_units"))
                st.success(f"Added organizational units configuration from file: {org_file.name}")
    
    return interpreter_configs if interpreter_configs else None

def main():
    st.title("CV Classification Pipeline")
    st.write("""
    Upload a CV to classify it using our AI-powered pipeline. The system will:
    1. Extract key information from your CV
    2. Classify your expertise areas, role levels, and organizational unit fit
    
    You can also customize the classification parameters using the customization options.
    """)
    
    # Add tabs for CV upload and customization
    tabs = st.tabs(["CV Upload & Processing", "Classification Customization", "Advanced File Interpretation"])
    
    # Tab 1: CV Upload & Processing
    with tabs[0]:
        # File uploader for CV text
        uploaded_file = st.file_uploader("Upload your CV (text format)", type=["txt", "pdf", "docx"])
        
        # Get CV text from either file or text area
        cv_text = ""
        file_name = None
        
        if uploaded_file is not None:
            file_name = uploaded_file.name
            try:
                cv_text = uploaded_file.getvalue().decode("utf-8")
                st.success(f"Successfully loaded CV from file: {file_name}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Text area for direct input
        st.write("#### Or paste your CV text below")
        text_input = st.text_area("CV Text", height=200)
        
        # Use text input if no file was uploaded or if text was entered
        if not cv_text and text_input:
            cv_text = text_input
            file_name = "manual_entry"
        
        # Display the CV text if available
        if cv_text:
            with st.expander("View CV text", expanded=False):
                st.text(cv_text)
        
        # Configuration file uploads
        st.write("#### Optional: Upload configuration files")
        st.info("You can upload pre-configured JSON files here, or create your own custom configuration in the **Classification Customization** tab or use **Advanced File Interpretation** for more complex configurations.")
        uploaded_config_files = st.file_uploader("Upload JSON configuration files", type=["json"], accept_multiple_files=True)
        
        config_files = []
        if uploaded_config_files:
            for config_file in uploaded_config_files:
                # Save the uploaded file temporarily
                temp_path = f"temp_config_{config_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(config_file.getbuffer())
                config_files.append(temp_path)
                st.success(f"Loaded configuration file: {config_file.name}")
        
        # Process button
        process_clicked = st.button("Process CV")
        
        if process_clicked:
            if not cv_text:
                st.warning("Please upload a file or paste CV text")
            else:
                # Collect custom configuration from the customization tab
                custom_config = {}
                if "expertise_config" in st.session_state and st.session_state.expertise_config:
                    custom_config["expertise"] = st.session_state.expertise_config
                
                if "role_levels_config" in st.session_state and st.session_state.role_levels_config:
                    custom_config["role_levels"] = st.session_state.role_levels_config
                
                if "org_units_config" in st.session_state and st.session_state.org_units_config:
                    custom_config["org_units"] = st.session_state.org_units_config
                
                # Get interpreter configurations
                interpreter_configs = None
                if "interpreter_configs" in st.session_state and st.session_state.interpreter_configs:
                    interpreter_configs = st.session_state.interpreter_configs

                print(f"interpreter_configs: {interpreter_configs}")
                
                # Process the CV
                results = process_cv_text(
                    cv_text, 
                    file_name, 
                    custom_config=custom_config, 
                    config_files=config_files,
                    interpreter_configs=interpreter_configs
                )
                
                if results:
                    display_results(results[-1])
                    
                    # Option to download results
                    st.download_button(
                        label="Download Results as JSON",
                        data=json.dumps(results[-1], indent=2),
                        file_name=f"cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                # Clean up temporary config files
                for temp_file in config_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                # Clean up temporary interpreter files
                if interpreter_configs:
                    for file_path, _, _ in interpreter_configs:
                        if os.path.exists(file_path):
                            os.remove(file_path)
    
    # Tab 2: Classification Customization
    with tabs[1]:
        st.write("""
        ## Classification Customization
        
        Use the options below to customize how CVs are classified. You can:
        - Define custom expertise categories
        - Define custom role levels
        - Define custom organizational units
        
        Your customizations will be applied when you process a CV.
        """)
        
        # Create expanders for each customization option
        with st.expander("Customize Expertise Categories", expanded=False):
            expertise_config = create_custom_expertise_config()
            if expertise_config:
                # Save to session state
                st.session_state.expertise_config = expertise_config
                
                # Option to download config
                st.download_button(
                    label="Download Expertise Configuration",
                    data=json.dumps({"expertise": expertise_config}, indent=2),
                    file_name=f"expertise_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with st.expander("Customize Role Levels", expanded=False):
            role_levels_config = create_custom_role_levels_config()
            if role_levels_config:
                # Save to session state
                st.session_state.role_levels_config = role_levels_config
                
                # Option to download config
                st.download_button(
                    label="Download Role Levels Configuration",
                    data=json.dumps({"role_levels": role_levels_config}, indent=2),
                    file_name=f"role_levels_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with st.expander("Customize Organizational Units", expanded=False):
            org_units_config = create_custom_org_units_config()
            if org_units_config:
                # Save to session state
                st.session_state.org_units_config = org_units_config
                
                # Option to download config
                st.download_button(
                    label="Download Org Units Configuration",
                    data=json.dumps({"org_units": org_units_config}, indent=2),
                    file_name=f"org_units_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Option to combine all configurations into a single file
        if (
            "expertise_config" in st.session_state or 
            "role_levels_config" in st.session_state or 
            "org_units_config" in st.session_state
        ):
            st.write("### Download Complete Configuration")
            
            combined_config = {}
            
            if "expertise_config" in st.session_state:
                combined_config["expertise"] = st.session_state.expertise_config
                
            if "role_levels_config" in st.session_state:
                combined_config["role_levels"] = st.session_state.role_levels_config
                
            if "org_units_config" in st.session_state:
                combined_config["org_units"] = st.session_state.org_units_config
            
            st.download_button(
                label="Download Complete Configuration",
                data=json.dumps(combined_config, indent=2),
                file_name=f"cv_classification_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Tab 3: Advanced File Interpretation
    with tabs[2]:
        st.write("""
        ## Advanced File Interpretation
        
        Use this feature to provide custom files with detailed descriptions of how the system should interpret them.
        Upload files containing your own categorization schemes, role level definitions, or organizational structures.
        
        For each uploaded file, provide a clear description of how it should be interpreted.
        The system will use an AI interpreter to convert your data into a format that the classification agents can use.
        """)
        
        interpreter_configs = create_interpreter_config()
        if interpreter_configs:
            # Save to session state
            st.session_state.interpreter_configs = interpreter_configs
            
            st.success("File interpretation configurations saved. These will be applied when you process a CV.")

if __name__ == "__main__":
    main()
