def extract_cv_sections(cv_data):
    """Extract relevant sections from CV JSON data"""
    work_experience_text = ""
    skills_text = ""
    education_text = ""
    
    # Extract work experience
    if "extracted_info" in cv_data and "work_experience" in cv_data["extracted_info"]:
        for job in cv_data["extracted_info"]["work_experience"]:
            company = job.get("company", "Unknown Company")
            title = job.get("title", "Unknown Position")
            duration = job.get("duration", "Unknown Duration")
            
            work_experience_text += f"Company: {company}\n"
            work_experience_text += f"Title: {title}\n"
            work_experience_text += f"Duration: {duration}\n"
            
            # Extract responsibilities
            if "responsibilities" in job and isinstance(job["responsibilities"], list):
                work_experience_text += "Responsibilities:\n"
                for resp in job["responsibilities"]:
                    work_experience_text += f"- {resp}\n"
            
            # Extract job-specific skills
            if "technical_skills" in job and isinstance(job["technical_skills"], list):
                work_experience_text += "Technical Skills:\n"
                for skill in job["technical_skills"]:
                    work_experience_text += f"- {skill}\n"
            
            if "soft_skills" in job and isinstance(job["soft_skills"], list):
                work_experience_text += "Soft Skills:\n"
                for skill in job["soft_skills"]:
                    work_experience_text += f"- {skill}\n"
            
            work_experience_text += "\n"
    
    # Extract extra skills
    if "extracted_info" in cv_data and "extra_skills" in cv_data["extracted_info"]:
        skills = cv_data["extracted_info"]["extra_skills"]
        
        if "Technical skills" in skills and isinstance(skills["Technical skills"], list):
            skills_text += "Technical Skills:\n"
            for skill in skills["Technical skills"]:
                skills_text += f"- {skill}\n"
        
        if "Soft skills" in skills and isinstance(skills["Soft skills"], list):
            skills_text += "Soft Skills:\n"
            for skill in skills["Soft skills"]:
                skills_text += f"- {skill}\n"
    
    # Extract education
    if "extracted_info" in cv_data and "education" in cv_data["extracted_info"]:
        for edu in cv_data["extracted_info"]["education"]:
            degree = edu.get("degree", "Unknown Degree")
            institution = edu.get("institution", "Unknown Institution")
            graduation_date = edu.get("graduation_date", "Unknown Date")
            
            education_text += f"Degree: {degree}\n"
            education_text += f"Institution: {institution}\n"
            education_text += f"Graduation Date: {graduation_date}\n\n"
    
    return {
        "work_experience": work_experience_text,
        "skills": skills_text,
        "education": education_text
    }
