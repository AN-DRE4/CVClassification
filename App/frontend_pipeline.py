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

results = []

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
        processor = CVProcessor(
            input_file="processed_cv_data.json", 
            output_dir="agents_results",
            custom_config=custom_config,
            config_files=config_files,
            interpreter_configs=interpreter_configs
        )
        try:
            results = processor.process_cvs(batch_size=1, save_interval=1, max_cvs=1)
            # Store orchestrator in session state for feedback handling
            st.session_state.orchestrator = processor.orchestrator
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
            for i, exp in enumerate(sorted_expertise):
                conf_pct = int(exp.get("confidence", 0) * 100)
                
                # Create a row with the category, percentage, and info button
                exp_col1, exp_col2 = st.columns([4, 1])
                
                with exp_col1:
                    st.write(f"{exp.get('category')} ({conf_pct}%)")
                    st.progress(exp.get("confidence", 0))
                
                with exp_col2:
                    # Use popover for justification display
                    with st.popover("ℹ️", help="View justification"):
                        st.write(f"**Justification for {exp.get('category')}:**")
                        st.write(exp.get("justification", "No justification provided"))
                        
                        # Show feedback adjustment info if available
                        if exp.get("feedback_adjustment") is not None:
                            original_conf = exp.get("original_confidence", exp.get("confidence", 0))
                            adjustment = exp.get("feedback_adjustment", 0)
                            st.info(f"Original confidence: {original_conf:.2f}, Feedback adjustment: {adjustment:+.2f}")
        else:
            st.write("No expertise areas identified")
    
    # Role Levels
    with col2:
        st.write("### Role Levels")
        role_levels = results.get("role_levels", {}).get("role_levels", [])
        if role_levels:
            # Sort by confidence
            sorted_roles = sorted(role_levels, key=lambda x: x.get("confidence", 0), reverse=True)
            for i, role in enumerate(sorted_roles):
                conf_pct = int(role.get("confidence", 0) * 100)
                
                # Create a row with the role info, percentage, and info button
                role_col1, role_col2 = st.columns([4, 1])
                
                with role_col1:
                    st.write(f"**{role.get('expertise')}:** {role.get('level')} ({conf_pct}%)")
                    st.progress(role.get("confidence", 0))
                
                with role_col2:
                    # Use popover for justification display
                    with st.popover("ℹ️", help="View justification"):
                        st.write(f"**Justification for {role.get('expertise')} - {role.get('level')}:**")
                        st.write(role.get("justification", "No justification provided"))
                        
                        # Show feedback adjustment info if available
                        if role.get("feedback_adjustment") is not None:
                            original_conf = role.get("original_confidence", role.get("confidence", 0))
                            adjustment = role.get("feedback_adjustment", 0)
                            st.info(f"Original confidence: {original_conf:.2f}, Feedback adjustment: {adjustment:+.2f}")
        else:
            st.write("No role levels identified")
    
    # Organizational Units
    with col3:
        st.write("### Organizational Units")
        org_units = results.get("org_unit", {}).get("org_units", [])
        if org_units:
            # Sort by confidence
            sorted_units = sorted(org_units, key=lambda x: x.get("confidence", 0), reverse=True)
            for i, unit in enumerate(sorted_units):
                conf_pct = int(unit.get("confidence", 0) * 100)
                
                # Create a row with the unit info, percentage, and info button
                unit_col1, unit_col2 = st.columns([4, 1])
                
                with unit_col1:
                    st.write(f"{unit.get('unit')} ({conf_pct}%)")
                    st.progress(unit.get("confidence", 0))
                
                with unit_col2:
                    # Use popover for justification display
                    with st.popover("ℹ️", help="View justification"):
                        st.write(f"**Justification for {unit.get('unit')}:**")
                        st.write(unit.get("justification", "No justification provided"))
                        
                        # Show feedback adjustment info if available
                        if unit.get("feedback_adjustment") is not None:
                            original_conf = unit.get("original_confidence", unit.get("confidence", 0))
                            adjustment = unit.get("feedback_adjustment", 0)
                            st.info(f"Original confidence: {original_conf:.2f}, Feedback adjustment: {adjustment:+.2f}")
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

def display_feedback_message(results):
    """Display a message directing users to the feedback tab instead of immediate feedback form"""
    st.write("---")
    st.write("### 📝 Feedback")
    st.info("🎯 **Help us improve our classification!** Go to the **Feedback Dashboard** tab to review and provide feedback on these results.")
    
    # Show quick stats if orchestrator is available
    if "orchestrator" in st.session_state:
        feedback_stats = st.session_state.orchestrator.get_feedback_stats()
        if feedback_stats["total_positive"] > 0 or feedback_stats["total_negative"] > 0:
            st.write(f"📊 **Current feedback:** {feedback_stats['total_positive']} positive, {feedback_stats['total_negative']} negative")

def validate_structured_feedback(feedback_text: str, result: Dict) -> Tuple[List[Dict], List[str]]:
    """Validate structured feedback and return parsed items and errors"""
    from cv_agents.utils.feedback_manager import FeedbackManager
    
    if not feedback_text or not feedback_text.strip():
        return [], []
    
    temp_feedback_manager = FeedbackManager()
    parsed_feedback = temp_feedback_manager.parse_structured_feedback(feedback_text)
    
    errors = []
    valid_feedback = []
    
    # Get available keys from the result
    expertise_keys = [exp.get('category', '').lower() for exp in result.get("expertise", {}).get("expertise", [])]
    role_keys = []
    for role in result.get("role_levels", {}).get("role_levels", []):
        expertise = role.get('expertise', '')
        level = role.get('level', '')
        role_keys.extend([expertise.lower(), level.lower(), f"{expertise}-{level}".lower()])
    org_keys = [unit.get('unit', '').lower() for unit in result.get("org_unit", {}).get("org_units", [])]
    
    for feedback_item in parsed_feedback:
        area = feedback_item['area']
        key = feedback_item['key'].lower()
        
        # Check if key exists in the classification results
        key_found = False
        if area == 'expertise' and key in expertise_keys:
            key_found = True
        elif area == 'role_level' and key in role_keys:
            key_found = True
        elif area == 'org_unit' and key in org_keys:
            key_found = True
        
        if key_found:
            valid_feedback.append(feedback_item)
        else:
            if area == 'expertise':
                errors.append(f"Expertise key '{feedback_item['key']}' not found. Available: {[exp.get('category', '') for exp in result.get('expertise', {}).get('expertise', [])]}")
            elif area == 'role_level':
                available_roles = []
                for role in result.get("role_levels", {}).get("role_levels", []):
                    expertise = role.get('expertise', '')
                    level = role.get('level', '')
                    available_roles.extend([expertise, level, f"{expertise}-{level}"])
                errors.append(f"Role level key '{feedback_item['key']}' not found. Available: {available_roles}")
            elif area == 'org_unit':
                errors.append(f"Org unit key '{feedback_item['key']}' not found. Available: {[unit.get('unit', '') for unit in result.get('org_unit', {}).get('org_units', [])]}")
    
    return valid_feedback, errors

def display_feedback_for_result(result, result_index):
    """Display feedback form for a specific result"""
    st.write(f"### CV Classification #{result_index + 1}")
    st.write(f"**Resume ID:** {result.get('resume_id', 'Unknown')}")
    st.write(f"**Processed:** {result.get('timestamp', 'Unknown')}")
    
    # Check if feedback was already provided for this CV
    feedback_key = f"feedback_saved_{result.get('resume_id', 'unknown')}"
    feedback_already_provided = st.session_state.get(feedback_key, False)
    
    if feedback_already_provided:
        st.success("✅ Thank you! Feedback has already been provided for this classification.")
        
        # Option to provide new feedback
        if st.button("Provide Additional Feedback", key=f"additional_feedback_{result_index}"):
            # Reset the feedback key to allow new feedback
            if feedback_key in st.session_state:
                del st.session_state[feedback_key]
            st.rerun()
        return
    
    # Display the results in a compact format
    with st.expander("View Classification Results", expanded=False):
        display_results(result)
    
    # Display feedback format instructions
    st.write("### 🎯 Provide Targeted Feedback")
    
    with st.expander("📋 How to Provide Structured Feedback", expanded=False):
        st.write("""
        **New Targeted Feedback Format:**
        
        Provide feedback on specific classification results using this format on each line:
        `area ::  key :: feedback`
        
        **Areas:**
        - `expertise` - for expertise area classifications
        - `role_level` - for role level classifications  
        - `org_unit` - for organizational unit classifications
        
        **Keys:**
        - For expertise: use the exact category name (e.g., `software_development`, `data_science`)
        - For role levels: use expertise name, level name, or "expertise-level" format
        - For org units: use the exact unit name (e.g., `engineering`, `marketing`)
        
        **Examples:**
        ```
        expertise :: software_development :: This classification is accurate
        role_level :: data_science-senior_level :: The level should be mid_level instead
        org_unit :: engineering :: Perfect match for this candidate
        expertise :: marketing :: This should not be classified as marketing
        ```
        
        **Benefits:**
        - Your feedback will be applied specifically to the items you mention
        - The system learns more precisely from your input
        - Future classifications of the same items will be improved
        """)
    
    # Use a form to collect feedback without page reloads
    with st.form(key=f"feedback_form_{result_index}", clear_on_submit=True):
        st.write("**How would you rate this classification overall?**")
        
        # Rating selection
        rating = st.selectbox(
            "Select your overall rating:",
            options=["Select an option", "👍 Good Classification", "👎 Needs Improvement"],
            index=0,
            key=f"rating_{result_index}"
        )
        
        # Detailed feedback with new format
        st.write("**Provide specific feedback using the structured format:**")
        
        # Get the actual results to help users with key names
        expertise_areas = result.get("expertise", {}).get("expertise", [])
        role_levels = result.get("role_levels", {}).get("role_levels", [])
        org_units = result.get("org_unit", {}).get("org_units", [])
        
        # Show available keys to help users
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if expertise_areas:
                st.write("**Available Expertise Areas:**")
                for exp in expertise_areas:
                    st.write(f"- `{exp.get('category', '')}`")
        
        with col2:
            if role_levels:
                st.write("**Available Role Levels:**")
                for role in role_levels:
                    expertise = role.get('expertise', '')
                    level = role.get('level', '')
                    st.write(f"- `{expertise}` or `{expertise}-{level}`")
        
        with col3:
            if org_units:
                st.write("**Available Org Units:**")
                for unit in org_units:
                    st.write(f"- `{unit.get('unit', '')}`")
        
        feedback_reason = st.text_area(
            "Structured feedback (one item per line):",
            help="Use format: area :: key :: feedback. Example: expertise :: software_development :: This classification is accurate",
            height=150,
            placeholder="expertise :: software_development :: This classification is accurate\nrole_level :: data_science-senior_level :: Should be mid_level instead\norg_unit :: engineering :: Perfect match",
            key=f"feedback_reason_{result_index}"
        )
        
        # Live validation of structured feedback
        if feedback_reason.strip():
            valid_feedback, errors = validate_structured_feedback(feedback_reason, result)
            
            if valid_feedback:
                st.success(f"✅ {len(valid_feedback)} valid feedback item(s) detected:")
                for item in valid_feedback:
                    st.write(f"  - **{item['area']}** → `{item['key']}`: {item['feedback']}")
            
            if errors:
                st.error("❌ Issues found:")
                for error in errors:
                    st.write(f"  - {error}")
        
        # Option for general feedback if structured format isn't used
        st.write("**Or provide general feedback (optional):**")
        general_feedback = st.text_area(
            "General comments:",
            help="Use this for overall comments that don't target specific classifications",
            height=80,
            placeholder="General comments about the overall classification quality...",
            key=f"general_feedback_{result_index}"
        )
        
        # Submit button
        submitted = st.form_submit_button("Submit Feedback", type="primary")
        
        if submitted:
            if rating == "Select an option":
                st.warning("Please select a rating before submitting.")
            else:
                # Determine rating type
                rating_type = "positive" if "Good" in rating else "negative"
                
                # Combine feedback text
                combined_feedback = ""
                if feedback_reason.strip():
                    combined_feedback = feedback_reason.strip()
                if general_feedback.strip():
                    if combined_feedback:
                        combined_feedback += "\n" + general_feedback.strip()
                    else:
                        combined_feedback = general_feedback.strip()
                
                if not combined_feedback:
                    combined_feedback = "No additional details provided"
                
                # Validate before submitting
                valid_feedback, errors = validate_structured_feedback(feedback_reason, result)
                
                if feedback_reason.strip() and errors:
                    st.error("Please fix the feedback format errors above before submitting.")
                    return
                
                # Save feedback
                result["user_feedback"] = {
                    "rating": rating_type,
                    "reason": combined_feedback,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Save to orchestrator
                if "orchestrator" in st.session_state:
                    st.session_state.orchestrator.add_feedback(result)
                    st.session_state[feedback_key] = True
                    
                    # Show success message
                    if rating_type == "positive":
                        st.success("👍 Thank you for your positive feedback!")
                    else:
                        st.success("Thank you for your feedback. We'll use this to improve our classification.")
                    
                    # Show what was parsed if structured feedback was used
                    if valid_feedback:
                        st.info("**Structured feedback successfully processed:**")
                        for feedback_item in valid_feedback:
                            st.write(f"- **{feedback_item['area']}** - '{feedback_item['key']}': {feedback_item['feedback']}")
                    
                    # Show feedback statistics
                    feedback_stats = st.session_state.orchestrator.get_feedback_stats()
                    if feedback_stats["total_positive"] > 0 or feedback_stats["total_negative"] > 0:
                        st.info(f"📊 Total feedback received: {feedback_stats['total_positive']} positive, {feedback_stats['total_negative']} negative")
                else:
                    st.warning("Could not save feedback - orchestrator not available")

def process_folder_for_frontend(folder_path, output_file):
    """Modified version of process_folder for frontend use"""
    # Load existing labeled data if it exists
    existing_data = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_data = json.load(f)

    # Get list of already processed resume IDs
    processed_ids = {item['resume_id'] for item in existing_data}

    # Load all resumes
    resumes = []
    supported_extensions = ['.txt', '.pdf', '.docx']
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if file has supported extension
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                file_path = os.path.join(root, file)
                if file_path not in processed_ids:  # Only add unprocessed resumes
                    try:
                        if file.lower().endswith('.txt'):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                resume_text = f.read()
                        else:
                            # For PDF and DOCX files, we'll just read them as text for now
                            # In a production environment, you'd want proper PDF/DOCX parsing
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                resume_text = f.read()
                        
                        resumes.append({
                            'id': file_path,
                            'text': resume_text
                        })
                    except Exception as e:
                        st.warning(f"Could not read file {file_path}: {str(e)}")

    # Process all remaining resumes
    processed_data = existing_data  # Start with existing data
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, resume in enumerate(resumes):
        progress = (i + 1) / len(resumes)
        progress_bar.progress(progress)
        status_text.text(f"Processing CV {i + 1} of {len(resumes)}: {os.path.basename(resume['id'])}")
        
        extracted_info = extract_resume_info(resume['text'])
        if extracted_info:
            processed_data.append({
                'resume_id': resume['id'],
                'resume_text': resume['text'],
                'extracted_info': extracted_info
            })

    # Save as silver data
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    progress_bar.progress(1.0)
    status_text.text(f"Completed! Processed {len(resumes)} new CVs, total: {len(processed_data)}")
    
    return processed_data

def process_cv_folder(folder_path, custom_config=None, config_files=None, interpreter_configs=None):
    """Process an entire folder of CVs through the pipeline"""
    import tempfile
    
    # Validate folder path
    if not os.path.exists(folder_path):
        st.error(f"Folder path does not exist: {folder_path}")
        return None
    
    # Check if folder contains any supported files
    supported_extensions = ['.txt', '.pdf', '.docx']
    cv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                cv_files.append(os.path.join(root, file))
    
    if not cv_files:
        st.warning(f"No supported CV files (.txt, .pdf, .docx) found in folder: {folder_path}")
        return None
    
    st.info(f"Found {len(cv_files)} CV files to process")
    
    # Step 1: Extract basic information from all CVs using zero-shot approach
    st.write("### Step 1: Extracting information from all CVs in folder")
    with st.spinner("Processing CVs with zero-shot approach..."):
        # Create a temporary file for the silver labeled data
        temp_silver_file = tempfile.mktemp(suffix='.json')
        
        try:
            # Use our modified process_folder function
            cv_data = process_folder_for_frontend(folder_path, temp_silver_file)
            
            if not cv_data:
                st.error("No CVs were successfully processed")
                return None
            
            st.success(f"Successfully processed {len(cv_data)} CVs from folder")
            
        except Exception as e:
            st.error(f"Error processing folder: {str(e)}")
            return None
        finally:
            # Clean up temp file
            if os.path.exists(temp_silver_file):
                os.remove(temp_silver_file)
    
    # Step 2: Process with classification agents
    st.write("### Step 2: Advanced CV Classification for all CVs")
    with st.spinner("Classifying all CVs with agents..."):
        try:
            # Create another temp file for the CV data input to CVProcessor
            temp_input_file = tempfile.mktemp(suffix='.json')
            with open(temp_input_file, 'w') as f:
                json.dump(cv_data, f)
            
            # Use CVProcessor for full classification
            processor = CVProcessor(
                input_file=temp_input_file,
                output_dir="batch_results",
                custom_config=custom_config,
                config_files=config_files,
                interpreter_configs=interpreter_configs
            )
            
            results = processor.process_cvs(
                batch_size=5,  # Smaller batch size for frontend
                save_interval=2, 
                max_cvs=len(cv_data)
            )
            
            # Store orchestrator in session state for feedback handling
            st.session_state.orchestrator = processor.orchestrator
            
            # Store batch results in session state for feedback
            if "batch_results" not in st.session_state:
                st.session_state.batch_results = []
            st.session_state.batch_results.extend(results)
            
            return results
            
        except Exception as e:
            st.error(f"Error during batch classification: {str(e)}")
            return None
        finally:
            # Clean up temp file
            if os.path.exists(temp_input_file):
                os.remove(temp_input_file)

def display_batch_results(results):
    """Display batch processing results in a formatted way"""
    st.write("## Batch Processing Results")
    st.write(f"**Total CVs Processed:** {len(results) + len(st.session_state.orchestrator.get_feedback_stats()['processed_cvs'])}")
    
    # Create summary statistics
    st.write("### Summary Statistics")
    
    # Count expertise areas
    expertise_counts = {}
    role_level_counts = {}
    org_unit_counts = {}
    
    for result in results:
        # Count expertise areas
        expertise_list = result.get("expertise", {}).get("expertise", [])
        for exp in expertise_list:
            category = exp.get("category")
            if category:
                expertise_counts[category] = expertise_counts.get(category, 0) + 1
        
        # Count role levels
        role_levels = result.get("role_levels", {}).get("role_levels", [])
        for role in role_levels:
            level_key = f"{role.get('expertise')} - {role.get('level')}"
            if level_key:
                role_level_counts[level_key] = role_level_counts.get(level_key, 0) + 1
        
        # Count org units
        org_units = result.get("org_unit", {}).get("org_units", [])
        for unit in org_units:
            unit_name = unit.get("unit")
            if unit_name:
                org_unit_counts[unit_name] = org_unit_counts.get(unit_name, 0) + 1
    
    # Display summary in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("#### Top Expertise Areas")
        sorted_expertise = sorted(expertise_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for category, count in sorted_expertise:
            st.write(f"**{category}:** {count} CVs")
    
    with col2:
        st.write("#### Top Role Levels")
        sorted_roles = sorted(role_level_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for role, count in sorted_roles:
            st.write(f"**{role}:** {count} CVs")
    
    with col3:
        st.write("#### Top Org Units")
        sorted_units = sorted(org_unit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for unit, count in sorted_units:
            st.write(f"**{unit}:** {count} CVs")
    
    st.write("---")
    
    # Individual CV results
    st.write("### Individual CV Results")
    
    # Option to select which CV to view in detail
    if len(results) > 0:
        selected_cv_index = st.selectbox(
            "Select a CV to view detailed results:",
            range(len(results)),
            format_func=lambda x: f"CV #{x + 1}: {results[x].get('resume_id', 'Unknown')}"
        )
        
        if selected_cv_index is not None:
            selected_result = results[selected_cv_index]
            
            # Display detailed results for selected CV
            with st.expander(f"Detailed Results for CV #{selected_cv_index + 1}", expanded=True):
                display_results(selected_result)
    
    # Download options
    st.write("### Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download all results
        st.download_button(
            label="Download All Results (JSON)",
            data=json.dumps(results, indent=2),
            file_name=f"batch_cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Download summary report
        summary_report = {
            "processing_date": datetime.now().isoformat(),
            "total_cvs": len(results),
            "summary_statistics": {
                "top_expertise_areas": dict(sorted_expertise),
                "top_role_levels": dict(sorted_roles),
                "top_org_units": dict(sorted_units)
            }
        }
        
        st.download_button(
            label="Download Summary Report (JSON)",
            data=json.dumps(summary_report, indent=2),
            file_name=f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    st.title("CV Classification Pipeline")
    
    # Initialize processed results in session state if not exists
    if "processed_results" not in st.session_state:
        st.session_state.processed_results = []
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🗂️ Navigation")
        st.write("### Pages")
        page = st.session_state.get("current_page", "🏠 Main Dashboard")
        if st.sidebar.button("🏠 Main Dashboard", use_container_width=True):
            page = "🏠 Main Dashboard"
            st.session_state.current_page = page
        if st.sidebar.button("📝 Feedback Dashboard", use_container_width=True):
            page = "📝 Feedback Dashboard"
            st.session_state.current_page = page
        
        # Display quick stats in sidebar if available
        if "orchestrator" in st.session_state:
            feedback_stats = st.session_state.orchestrator.get_feedback_stats()
            if feedback_stats["total_positive"] > 0 or feedback_stats["total_negative"] > 0:
                st.write("---")
                st.write("**📊 Feedback Stats:**")
                st.write(f"👍 Positive: {feedback_stats['total_positive']}")
                st.write(f"👎 Negative: {feedback_stats['total_negative']}")
                total_feedback = feedback_stats['total_positive'] + feedback_stats['total_negative']
                if total_feedback > 0:
                    positive_rate = (feedback_stats['total_positive'] / total_feedback) * 100
                    st.write(f"✅ Positive Rate: {positive_rate:.1f}%")
    
    # Main Dashboard Page
    if page == "🏠 Main Dashboard":
        st.write("""
        Upload a CV to classify it using our AI-powered pipeline. The system will:
        1. Extract key information from your CV
        2. Classify your expertise areas, role levels, and organizational unit fit
        
        You can also customize the classification parameters using the customization options.
        """)
        
        # Add tabs for CV upload and customization (original tabs 1-3)
        tabs = st.tabs(["CV Upload & Processing", "Batch Processing", "Classification Customization", "Advanced File Interpretation"])
        
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
                    
                    # Process the CV
                    result = process_cv_text(
                        cv_text, 
                        file_name, 
                        custom_config=custom_config, 
                        config_files=config_files,
                        interpreter_configs=interpreter_configs
                    )
                    
                    if result:
                        # Store the result in session state
                        st.session_state.processed_results.append(result[-1])
                        
                        # Display results
                        display_results(result[-1])
                        
                        # Option to download results
                        st.download_button(
                            label="Download Results as JSON",
                            data=json.dumps(result[-1], indent=2),
                            file_name=f"cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        
                        # Display feedback message instead of immediate feedback form
                        display_feedback_message(result[-1])
                                                               
                    # Clean up temporary config files
                    for temp_file in config_files:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    
                    # Clean up temporary interpreter files
                    if interpreter_configs:
                        for file_path, _, _ in interpreter_configs:
                            if os.path.exists(file_path):
                                os.remove(file_path)
        
        # Tab 2: Batch Processing
        with tabs[1]:
            st.write("""
            ## Batch Processing
            
            Process multiple CVs at once by specifying a folder containing CV files.
            
            **Supported file formats:** .txt, .pdf, .docx
            """)
            
            # Folder input section
            st.write("### Select Folder")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                folder_path = st.text_input(
                    "Enter the path to the folder containing CVs",
                    placeholder="e.g., /path/to/cv/folder or C:\\path\\to\\cv\\folder",
                    help="Enter the full path to the folder containing CV files"
                )
            
            with col2:
                st.write("")  # Add some spacing
                st.write("")  # Add some spacing
                if st.button("📁 Browse", help="You can manually enter the path in the text field"):
                    st.info("💡 **Tip:** Copy and paste the folder path from your file explorer into the text field above.")
            
            # Configuration section
            st.write("### Optional: Configuration")
            st.info("You can use the same customization options from the **Classification Customization** and **Advanced File Interpretation** tabs.")
            
            # Configuration file uploads for batch processing
            uploaded_batch_config_files = st.file_uploader(
                "Upload JSON configuration files for batch processing", 
                type=["json"], 
                accept_multiple_files=True,
                key="batch_config_files"
            )
            
            batch_config_files = []
            if uploaded_batch_config_files:
                for config_file in uploaded_batch_config_files:
                    # Save the uploaded file temporarily
                    temp_path = f"temp_batch_config_{config_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(config_file.getbuffer())
                    batch_config_files.append(temp_path)
                    st.success(f"Loaded configuration file: {config_file.name}")
            
            # Process button
            if st.button("🚀 Process Folder", type="primary"):
                if not folder_path:
                    st.warning("Please enter a folder path")
                elif not os.path.exists(folder_path):
                    st.error("The specified folder path does not exist. Please check the path and try again.")
                else:
                    # Collect custom configuration from session state
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
                    
                    # Process the folder
                    result = process_cv_folder(
                        folder_path,
                        custom_config=custom_config,
                        config_files=batch_config_files,
                        interpreter_configs=interpreter_configs
                    )
                    
                    if result:
                        # Display batch results
                        display_batch_results(result)
                        
                        # Show feedback message
                        st.write("---")
                        st.write("### 📝 Feedback")
                        st.info("🎯 **Help us improve our classification!** Go to the **Feedback Dashboard** to review and provide feedback on these batch results.")
                    
                    # Clean up temporary config files
                    for temp_file in batch_config_files:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
            
            # Show previous batch results if available
            if "batch_results" in st.session_state and st.session_state.batch_results:
                st.write("---")
                st.write("### Previous Batch Results")
                
                if st.button("Show Previous Batch Results"):
                    display_batch_results(st.session_state.batch_results)
                
                if st.button("Clear Previous Batch Results", type="secondary"):
                    st.session_state.batch_results = []
                    st.success("Previous batch results cleared!")
                    st.rerun()
        
        # Tab 2: Classification Customization
        with tabs[2]:
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
        with tabs[3]:
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

    # Feedback Dashboard Page  
    elif page == "📝 Feedback Dashboard":
        st.write("""
        ## Feedback Dashboard
        
        Review all processed CV classifications and provide feedback to help improve the system.
        """)
        
        # Combine individual and batch results
        all_results = []
        if st.session_state.processed_results:
            all_results.extend(st.session_state.processed_results)
        if "batch_results" in st.session_state and st.session_state.batch_results:
            all_results.extend(st.session_state.batch_results)
        
        # Check if there are any processed results
        if not all_results:
            st.info("No CV classifications available for feedback. Please process CVs first in the **Main Dashboard**.")
            return
        
        # Display overall statistics if orchestrator is available
        if "orchestrator" in st.session_state:
            feedback_stats = st.session_state.orchestrator.get_feedback_stats()
            
            # Display overall statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total CVs", len(all_results))
            
            with col2:
                st.metric("Positive Feedback", feedback_stats.get('total_positive', 0))
            
            with col3:
                st.metric("Negative Feedback", feedback_stats.get('total_negative', 0))
            
            with col4:
                total_feedback = feedback_stats.get('total_positive', 0) + feedback_stats.get('total_negative', 0)
                if total_feedback > 0:
                    positive_rate = (feedback_stats.get('total_positive', 0) / total_feedback) * 100
                    st.metric("Positive Rate", f"{positive_rate:.1f}%")
                else:
                    st.metric("Positive Rate", "N/A")
            
            # Show last updated
            if feedback_stats.get('last_updated'):
                st.write(f"**Last updated:** {feedback_stats['last_updated']}")
        
        st.write("---")
        
        # Filter options
        st.write("### Filter Results")
        col1, col2 = st.columns(2)
        
        with col1:
            result_type_filter = st.selectbox(
                "Filter by source:",
                ["All Results", "Individual CVs", "Batch Processed CVs"],
                index=0
            )
        
        with col2:
            feedback_filter = st.selectbox(
                "Filter by feedback status:",
                ["All", "With Feedback", "Without Feedback"],
                index=0
            )
        
        # Apply filters
        filtered_results = all_results.copy()
        
        if result_type_filter == "Individual CVs":
            filtered_results = st.session_state.processed_results or []
        elif result_type_filter == "Batch Processed CVs":
            filtered_results = st.session_state.get("batch_results", [])
        
        if feedback_filter == "With Feedback":
            filtered_results = [r for r in filtered_results if r.get("user_feedback")]
        elif feedback_filter == "Without Feedback":
            filtered_results = [r for r in filtered_results if not r.get("user_feedback")]
        
        # Display filtered results for feedback
        st.write(f"### Filtered Results ({len(filtered_results)} total)")
        
        if not filtered_results:
            st.info("No results match the selected filters.")
            return
        
        # Option to select which CV to provide feedback on
        if len(filtered_results) > 1:
            selected_cv = st.selectbox(
                "Select a CV to provide feedback on:",
                range(len(filtered_results)),
                format_func=lambda x: f"CV #{x + 1}: {os.path.basename(filtered_results[x].get('resume_id', 'Unknown'))}"
            )
        else:
            selected_cv = 0
        
        if selected_cv is not None and selected_cv < len(filtered_results):
            result = filtered_results[selected_cv]
            display_feedback_for_result(result, selected_cv)
        
        st.write("---")

if __name__ == "__main__":
    main()
