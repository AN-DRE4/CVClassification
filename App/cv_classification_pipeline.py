import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import re
import os
from datasets import Dataset
from collections import defaultdict

# Define expertise categories (based on technical skills)
EXPERTISE_CATEGORIES = {
    'software_development': [
        'python', 'java', 'c#', 'asp.net', 'javascript', 'html', 'css', 
        'react', 'angular', 'vue', 'node.js', '.net', 'c++', 'php'
    ],
    'data_engineering': [
        'sql', 'etl', 'hadoop', 'spark', 'aws', 'data warehouse', 'azure', 
        'databricks', 'snowflake', 'redshift', 'big data', 'oracle', 'ms sql server'
    ],
    'data_science': [
        'machine learning', 'deep learning', 'ai', 'artificial intelligence', 
        'python', 'r', 'statistics', 'tensorflow', 'pytorch', 'keras', 'data analysis'
    ],
    'devops': [
        'jenkins', 'docker', 'kubernetes', 'aws', 'azure', 'devops', 'ci/cd',
        'terraform', 'ansible', 'linux', 'bash', 'shell', 'deployment'
    ],
    'cybersecurity': [
        'security', 'penetration testing', 'ethical hacking', 'network security',
        'firewall', 'encryption', 'vpn', 'compliance', 'risk assessment'
    ],
    'marketing': [
        'seo', 'content writing', 'google analytics', 'social media', 
        'digital marketing', 'content strategy', 'advertising'
    ],
    'finance': [
        'excel', 'accounting', 'financial analysis', 'financial reporting',
        'banking', 'reconciliation', 'tax', 'audit', 'tally'
    ],
    'management': [
        'project management', 'team lead', 'management', 'leadership',
        'agile', 'scrum', 'product owner', 'release management', 'change management'
    ]
}

# Define role levels based on job titles
ROLE_LEVELS = {
    'entry_level': [
        'associate', 'junior', 'intern', 'trainee', 'assistant'
    ],
    'mid_level': [
        'engineer', 'developer', 'analyst', 'consultant', 'specialist', 'administrator'
    ],
    'senior_level': [
        'senior', 'lead', 'architect', 'principal', 'staff'
    ],
    'management': [
        'manager', 'head', 'director', 'chief', 'vp', 'president'
    ]
}

# Define role level hierarchy from highest to lowest
ROLE_LEVEL_HIERARCHY = ['management', 'senior_level', 'mid_level', 'entry_level', 'unknown']

# Define organizational units based on job roles and responsibilities
ORG_UNITS = {
    'engineering': [
        'software', 'development', 'programming', 'coding', 'testing', 'qa',
        'engineering', 'devops', 'developer', 'architect'
    ],
    'data': [
        'data', 'analytics', 'business intelligence', 'reporting', 'database',
        'data science', 'machine learning', 'ai', 'statistics'
    ],
    'marketing_sales': [
        'marketing', 'sales', 'seo', 'social media', 'advertising', 'content',
        'campaign', 'lead generation', 'customer acquisition'
    ],
    'finance_accounting': [
        'finance', 'accounting', 'financial', 'audit', 'tax', 'banking',
        'investment', 'treasury', 'reconciliation'
    ],
    'operations': [
        'operations', 'project management', 'process', 'improvement', 'delivery',
        'production', 'supply chain', 'logistics'
    ],
    'customer_service': [
        'customer', 'support', 'service', 'helpdesk', 'client', 'account management'
    ],
    'hr': [
        'hr', 'human resources', 'recruitment', 'talent', 'hiring', 'personnel',
        'training', 'development', 'compensation', 'benefits'
    ]
}

class CVClassifier:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.expertise_model = None
        self.role_level_model = None
        self.org_unit_model = None
        self.expertise_binarizer = MultiLabelBinarizer()
        self.role_level_binarizer = MultiLabelBinarizer()
        self.org_unit_binarizer = MultiLabelBinarizer()
        self.role_level_models = {}
        self.role_level_models_binarizers = {}
        
    def load_data(self, file_path):
        """Load labeled resume data from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} resumes")
        return data
    
    def preprocess_resumes(self, resumes):
        """Extract features and prepare data for training"""
        processed_data = []
        
        for resume in resumes:
            extracted_info = resume.get('extracted_info', {})
            
            # Extract text representation
            all_text = self._extract_resume_text(extracted_info)
            
            # Determine expertise categories by work experience
            expertise_by_job = self._determine_expertise_by_job(extracted_info)
            
            # Get overall expertise categories (flattened)
            expertise = list(set(exp for job_exp in expertise_by_job.values() for exp in job_exp))
            if not expertise:
                expertise = ['unknown']
            
            # Determine role level for each expertise category
            role_level_by_expertise = self._determine_role_level_by_expertise(extracted_info, expertise_by_job)
            
            # For each expertise, keep only the highest role level
            for exp in role_level_by_expertise:
                role_levels = role_level_by_expertise[exp]
                highest_level = 'unknown'
                for level in ROLE_LEVEL_HIERARCHY:
                    if level in role_levels:
                        highest_level = level
                        break
                role_level_by_expertise[exp] = [highest_level]
            
            # Determine organizational unit
            org_unit = self._determine_org_unit(extracted_info)
            
            processed_data.append({
                'resume_id': resume.get('resume_id', ''),
                'text': all_text,
                'expertise': expertise,
                'expertise_by_job': expertise_by_job,
                'role_level_by_expertise': role_level_by_expertise,
                'org_unit': org_unit
            })
        
        return processed_data
    
    def _extract_resume_text(self, extracted_info):
        """Extract and concatenate text from resume sections"""
        texts = []
        
        # Extract education info
        if 'education' in extracted_info:
            for edu in extracted_info['education']:
                edu_text = f"{edu.get('degree', '')} {edu.get('institution', '')} {edu.get('graduation_date', '')}"
                texts.append(edu_text)
        
        # Extract work experience including job-specific skills
        if 'work_experience' in extracted_info:
            for exp in extracted_info['work_experience']:
                exp_text = f"{exp.get('company', '')} {exp.get('title', '')} "
                
                # Add responsibilities
                if 'responsibilities' in exp and isinstance(exp['responsibilities'], list):
                    resp_text = ' '.join(exp.get('responsibilities', []))
                    exp_text += resp_text
                    
                # Add technical skills specific to this job
                if 'technical_skills' in exp and isinstance(exp['technical_skills'], list):
                    tech_skills = ' '.join(exp.get('technical_skills', []))
                    exp_text += f" Technical skills: {tech_skills}"
                    
                # Add soft skills specific to this job
                if 'soft_skills' in exp and isinstance(exp['soft_skills'], list):
                    soft_skills = ' '.join(exp.get('soft_skills', []))
                    exp_text += f" Soft skills: {soft_skills}"
                    
                texts.append(exp_text)
        
        # Extract extra skills that aren't associated with specific jobs
        if 'extra_skills' in extracted_info:
            if 'Technical skills' in extracted_info['extra_skills'] and isinstance(extracted_info['extra_skills']['Technical skills'], list):
                extra_tech = ' '.join(extracted_info['extra_skills']['Technical skills'])
                texts.append(f"Extra technical skills: {extra_tech}")
                
            if 'Soft skills' in extracted_info['extra_skills'] and isinstance(extracted_info['extra_skills']['Soft skills'], list):
                extra_soft = ' '.join(extracted_info['extra_skills']['Soft skills'])
                texts.append(f"Extra soft skills: {extra_soft}")
        
        return ' '.join(texts)
    
    def _map_skill_to_expertise(self, skill):
        """Map a skill to corresponding expertise categories"""
        if not isinstance(skill, str):
            return []
            
        skill_lower = skill.lower()
        matched_expertise = []
        
        for category, keywords in EXPERTISE_CATEGORIES.items():
            for keyword in keywords:
                if keyword.lower() in skill_lower or skill_lower in keyword.lower():
                    matched_expertise.append(category)
                    break
        
        return matched_expertise
    
    def _determine_expertise_by_job(self, extracted_info):
        """Determine expertise categories for each job based on skills and responsibilities"""
        expertise_by_job = {}
        
        # Process work experience entries
        if 'work_experience' in extracted_info:
            for i, exp in enumerate(extracted_info['work_experience']):
                job_id = f"{exp.get('company', 'unknown')}_{i}"
                job_title = exp.get('title', '').lower()
                job_expertise = []
                
                # Extract skills for this job
                tech_skills = []
                if 'technical_skills' in exp and isinstance(exp['technical_skills'], list):
                    tech_skills = [s.lower() for s in exp['technical_skills'] if isinstance(s, str)]
                
                # Extract responsibilities
                responsibilities = ""
                if 'responsibilities' in exp and isinstance(exp['responsibilities'], list):
                    responsibilities = ' '.join([r.lower() for r in exp['responsibilities'] if isinstance(r, str)])
                
                # Match skills to expertise categories
                for skill in tech_skills:
                    matched = self._map_skill_to_expertise(skill)
                    job_expertise.extend(matched)
                
                # Match job title and responsibilities to expertise categories
                for category, keywords in EXPERTISE_CATEGORIES.items():
                    if any(keyword.lower() in job_title for keyword in keywords):
                        job_expertise.append(category)
                        continue
                        
                    if any(keyword.lower() in responsibilities for keyword in keywords):
                        job_expertise.append(category)
                
                # Remove duplicates
                job_expertise = list(set(job_expertise))
                
                # If no expertise found, check if there are generic indicators
                if not job_expertise:
                    if 'developer' in job_title or 'engineer' in job_title:
                        job_expertise.append('software_development')
                    elif 'analyst' in job_title:
                        job_expertise.append('data_engineering')
                    elif 'manager' in job_title or 'director' in job_title:
                        job_expertise.append('management')
                
                # Store expertise for this job
                expertise_by_job[job_id] = job_expertise if job_expertise else ['unknown']
        
        # Process extra skills (not associated with specific jobs)
        extra_expertise = []
        if 'extra_skills' in extracted_info and 'Technical skills' in extracted_info['extra_skills']:
            extra_tech_skills = extracted_info['extra_skills']['Technical skills']
            if isinstance(extra_tech_skills, list):
                for skill in extra_tech_skills:
                    if isinstance(skill, str):
                        matched = self._map_skill_to_expertise(skill)
                        extra_expertise.extend(matched)
        
        # Add extra expertise to a special category
        if extra_expertise:
            expertise_by_job['extra_skills'] = list(set(extra_expertise))
        
        return expertise_by_job
    
    def _determine_role_level_by_expertise(self, extracted_info, expertise_by_job):
        """Determine role level for each expertise category based on job titles and experience"""
        # Initialize with empty dictionary
        role_levels_by_expertise = {}
        
        # Process work experience to determine role levels
        if 'work_experience' in extracted_info:
            for i, exp in enumerate(extracted_info['work_experience']):
                job_id = f"{exp.get('company', 'unknown')}_{i}"
                
                # Skip if job has no associated expertise
                if job_id not in expertise_by_job or not expertise_by_job[job_id]:
                    continue
                
                # Get job details
                job_title = exp.get('title', '').lower()
                duration = exp.get('duration', 0)
                if not isinstance(duration, (int, float)) or duration == 'N/A':
                    duration = 0
                
                # Determine role level from job title
                role_level = 'unknown'
                # Find all matching role levels
                matched_levels = []
                
                # Find all matching role levels
                for level, keywords in ROLE_LEVELS.items():
                    if any(keyword.lower() in job_title for keyword in keywords):
                        matched_levels.append(level)
                
                # Select highest level based on hierarchy
                if matched_levels:
                    for level in ROLE_LEVEL_HIERARCHY:
                        if level in matched_levels:
                            role_level = level
                            break
                
                # If no role level found from title, use duration as a hint
                if role_level == 'unknown':
                    if duration >= 5:
                        role_level = 'senior_level'
                    elif duration >= 2:
                        role_level = 'mid_level'
                    elif duration > 0:
                        role_level = 'entry_level'
                
                # Bump up role level based on duration
                if duration >= 5:
                    # Find current level index in hierarchy
                    current_idx = ROLE_LEVEL_HIERARCHY.index(role_level) if role_level in ROLE_LEVEL_HIERARCHY else -1
                    
                    # Can only bump up to senior_level maximum
                    if current_idx > ROLE_LEVEL_HIERARCHY.index('senior_level'):
                        # Bump up one level
                        role_level = ROLE_LEVEL_HIERARCHY[current_idx - 1]
                
                # Associate this role level with each expertise category for this job
                for expertise in expertise_by_job[job_id]:
                    if expertise not in role_levels_by_expertise:
                        role_levels_by_expertise[expertise] = []
                    
                    if role_level not in role_levels_by_expertise[expertise]:
                        role_levels_by_expertise[expertise].append(role_level)
        
        # For expertise categories without role levels, mark as unknown
        all_expertise = set(exp for job_exps in expertise_by_job.values() for exp in job_exps)
        for expertise in all_expertise:
            if expertise not in role_levels_by_expertise or not role_levels_by_expertise[expertise]:
                role_levels_by_expertise[expertise] = ['unknown']
        
        return role_levels_by_expertise
    
    def _determine_org_unit(self, extracted_info):
        """Determine organizational unit based on job titles, skills, and responsibilities"""
        org_units = []
        
        # Process work experience to determine org units
        if 'work_experience' in extracted_info:
            for exp in extracted_info['work_experience']:
                # Gather text from job title and responsibilities
                job_text = exp.get('title', '').lower()
                
                if 'responsibilities' in exp and isinstance(exp['responsibilities'], list):
                    job_text += ' ' + ' '.join([r.lower() for r in exp['responsibilities'] if isinstance(r, str)])
                
                # Check technical skills
                if 'technical_skills' in exp and isinstance(exp['technical_skills'], list):
                    job_text += ' ' + ' '.join([s.lower() for s in exp['technical_skills'] if isinstance(s, str)])
                
                # Match to organizational units
                for unit, keywords in ORG_UNITS.items():
                    if any(keyword.lower() in job_text for keyword in keywords):
                        org_units.append(unit)
        
        # Check extra skills
        if 'extra_skills' in extracted_info and 'Technical skills' in extracted_info['extra_skills']:
            extra_tech = ' '.join(extracted_info['extra_skills']['Technical skills'])
            for unit, keywords in ORG_UNITS.items():
                if any(keyword.lower() in extra_tech.lower() for keyword in keywords):
                    org_units.append(unit)
        
        # Remove duplicates
        org_units = list(set(org_units))
        
        return org_units if org_units else ['unknown']
    
    def prepare_for_training(self, processed_data):
        """Prepare data for model training"""
        # Extract text
        texts = [item['text'] for item in processed_data]
        
        # One-hot encode the expertise labels
        expertise_labels = [item['expertise'] for item in processed_data]
        expertise_encoded = self.expertise_binarizer.fit_transform(expertise_labels)
        
        # For role levels, create separate datasets for each expertise category
        role_level_datasets = {}
        for category in set(cat for item in processed_data for cat in item['expertise'] if cat != 'unknown'):
            # Get role levels for this category from each resume that has this expertise
            category_role_levels = []
            category_texts = []
            
            for item in processed_data:
                if category in item['expertise']:
                    category_role_levels.append(item['role_level_by_expertise'].get(category, ['unknown']))
                    category_texts.append(item['text'])
            
            if category_role_levels:
                # Create binarizer for this category
                category_binarizer = MultiLabelBinarizer()
                category_encoded = category_binarizer.fit_transform(category_role_levels)
                
                # Create dataset with float32 labels
                category_dataset = Dataset.from_dict({
                    'text': category_texts,
                    'label': category_encoded.astype(np.float32).tolist()  # Convert to float32
                })
                
                role_level_datasets[category] = {
                    'dataset': category_dataset,
                    'binarizer': category_binarizer
                }
        
        # Org unit labels
        org_unit_labels = [item['org_unit'] for item in processed_data]
        org_unit_encoded = self.org_unit_binarizer.fit_transform(org_unit_labels)
        
        # Create org unit dataset with float32 labels
        org_unit_dataset = Dataset.from_dict({
            'text': texts,
            'label': org_unit_encoded.astype(np.float32).tolist()  # Convert to float32
        })
        
        # Create expertise dataset with float32 labels
        expertise_dataset = Dataset.from_dict({
            'text': texts,
            'label': expertise_encoded.astype(np.float32).tolist()  # Convert to float32
        })
        
        return {
            'expertise': expertise_dataset,
            'role_level_by_expertise': role_level_datasets,
            'org_unit': org_unit_dataset
        }
    
    def init_models(self, datasets):
        """Initialize BERT-based models for each classification task"""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # Create model for expertise classification
        self.expertise_model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.expertise_binarizer.classes_),
            problem_type="multi_label_classification"
        )
        
        # Create models for role level classification (one per expertise category)
        self.role_level_models = {}
        for category, category_data in datasets['role_level_by_expertise'].items():
            self.role_level_models[category] = BertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(category_data['binarizer'].classes_),
                problem_type="multi_label_classification"
            )
        
        # Create model for org unit classification
        self.org_unit_model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.org_unit_binarizer.classes_),
            problem_type="multi_label_classification"
        )
    
    def tokenize_function(self, examples):
        """Tokenization function for BERT model"""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    def train_models(self, datasets, output_dir="./cv_classifier_models"):
        """Train classification models"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models
        self.init_models(datasets)
        
        # Train expertise model
        print("\nTraining expertise model...")
        self._train_model(
            datasets['expertise'], 
            self.expertise_model, 
            f"{output_dir}/expertise"
        )
        
        # Train role level models for each expertise category
        for category, category_data in datasets['role_level_by_expertise'].items():
            # Check if we have enough samples to split into train/test
            dataset_size = len(category_data['dataset'])
            if dataset_size <= 1:
                print(f"Skipping training for {category} - insufficient data (only {dataset_size} sample)")
                continue
            
            # For very small datasets, adjust test_size
            test_size = 0.2
            if dataset_size < 5:
                test_size = 1/dataset_size  # Just 1 sample for testing
                print(f"Warning: Small dataset for {category} ({dataset_size} samples). Using minimal test split.")
            
            print(f"\nTraining role level model for {category}...")
            os.makedirs(f"{output_dir}/role_level/{category}", exist_ok=True)
            self._train_model(
                category_data['dataset'],
                self.role_level_models[category],
                f"{output_dir}/role_level/{category}",
                test_size=test_size
            )
            
            # Save binarizer
            with open(f"{output_dir}/role_level/{category}/binarizer.json", 'w') as f:
                json.dump({
                    'classes_': category_data['binarizer'].classes_.tolist()
                }, f)
        
        # Train org unit model
        print("\nTraining organizational unit model...")
        self._train_model(
            datasets['org_unit'], 
            self.org_unit_model, 
            f"{output_dir}/org_unit"
        )
    
    def _train_model(self, dataset, model, output_dir, test_size=0.2):
        """Train a specific classification model"""
        # Get dataset size
        dataset_size = len(dataset)
        print(f"Training model with {dataset_size} samples")
        
        # Split dataset into train/test
        dataset_dict = dataset.train_test_split(test_size=test_size)
        
        # Tokenize datasets
        tokenized_datasets = dataset_dict.map(self.tokenize_function, batched=True)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            weight_decay=0.01,
            save_strategy="epoch",
            eval_strategy="epoch",  # Changed to match current API
            load_best_model_at_end=True,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            # Removed deprecated parameters
        )
        
        # Train model
        trainer.train()
        
        # Save model and tokenizer
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def load_trained_models(self, model_dir="./cv_classifier_models"):
        """Load trained models for inference"""
        # Load expertise model
        self.expertise_model = BertForSequenceClassification.from_pretrained(f"{model_dir}/expertise")
        self.tokenizer = BertTokenizer.from_pretrained(f"{model_dir}/expertise")
        
        # Load role level models (we'd need to know which expertise categories in advance)
        self.role_level_models = {}
        role_level_dirs = [d for d in os.listdir(f"{model_dir}/role_level") if os.path.isdir(os.path.join(f"{model_dir}/role_level", d))]
        
        for category in role_level_dirs:
            # Load model
            self.role_level_models[category] = BertForSequenceClassification.from_pretrained(
                f"{model_dir}/role_level/{category}"
            )
            
            # Load binarizer
            with open(f"{model_dir}/role_level/{category}/binarizer.json", 'r') as f:
                binarizer_data = json.load(f)
            
            category_binarizer = MultiLabelBinarizer()
            category_binarizer.classes_ = np.array(binarizer_data['classes_'])
            self.role_level_models_binarizers[category] = category_binarizer
        
        # Load org unit model
        self.org_unit_model = BertForSequenceClassification.from_pretrained(f"{model_dir}/org_unit")
        
        # Load label binarizers (would need to save/load these in practice)
        # For now we'll just use the existing ones from training
    
    def predict(self, resume_text):
        """Predict classifications for a new resume"""
        # Tokenize text
        inputs = self.tokenizer(
            resume_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Get predictions from expertise model
        with torch.no_grad():
            expertise_outputs = self.expertise_model(**inputs)
            org_unit_outputs = self.org_unit_model(**inputs)
        
        # Convert logits to predictions
        expertise_preds = torch.sigmoid(expertise_outputs.logits) > 0.5
        org_unit_preds = torch.sigmoid(org_unit_outputs.logits) > 0.5
        
        # Convert to class names
        expertise_classes = [self.expertise_binarizer.classes_[i] for i, pred in enumerate(expertise_preds[0]) if pred]
        org_unit_classes = [self.org_unit_binarizer.classes_[i] for i, pred in enumerate(org_unit_preds[0]) if pred]
        
        # Get role level predictions for each predicted expertise
        role_level_by_expertise = {}
        for category in expertise_classes:
            if category in self.role_level_models:
                with torch.no_grad():
                    category_outputs = self.role_level_models[category](**inputs)
                
                category_preds = torch.sigmoid(category_outputs.logits) > 0.5
                category_binarizer = self.role_level_models_binarizers[category]
                role_level_by_expertise[category] = [
                    category_binarizer.classes_[i] for i, pred in enumerate(category_preds[0]) if pred
                ]
            else:
                role_level_by_expertise[category] = ['unknown']
        
        return {
            'expertise': expertise_classes,
            'role_level_by_expertise': role_level_by_expertise,
            'org_unit': org_unit_classes
        }


def main():
    """Main function to demonstrate the pipeline"""
    # Initialize classifier
    cv_classifier = CVClassifier()
    
    # Load data
    resumes = cv_classifier.load_data('silver_labeled_resumes.json')
    
    # Preprocess data
    processed_resumes = cv_classifier.preprocess_resumes(resumes)
    
    # Print some statistics
    expertise_counts = defaultdict(list)
    job_expertise_counts = defaultdict(lambda: defaultdict(list))
    role_level_by_expertise_counts = defaultdict(lambda: defaultdict(list))
    org_unit_counts = defaultdict(list)
    
    for resume in processed_resumes:
        resume_id = resume['resume_id']
        
        # Track overall expertise
        for exp in resume['expertise']:
            expertise_counts[exp].append(resume_id)
        
        # Track expertise by job
        for job_id, job_expertise in resume['expertise_by_job'].items():
            for exp in job_expertise:
                job_expertise_counts[job_id][exp].append(resume_id)
        
        # Track role level by expertise
        for exp, role_levels in resume['role_level_by_expertise'].items():
            for role in role_levels:
                role_level_by_expertise_counts[exp][role].append(resume_id)
        
        # Track org unit
        for unit in resume['org_unit']:
            org_unit_counts[unit].append(resume_id)
    
    print("\nOverall Expertise Distribution:")
    for exp, resume_ids in expertise_counts.items():
        print(f"{exp}: {len(resume_ids)}")
        print(f"  Resumes: {', '.join(resume_ids)}")
    
    print("\nExpertise by Job:")
    for job_id, expertise_dict in job_expertise_counts.items():
        print(f"\n  Job: {job_id}")
        for exp, resume_ids in expertise_dict.items():
            print(f"    {exp}: {len(resume_ids)}")
            print(f"      Resumes: {', '.join(resume_ids)}")
    
    print("\nRole Level Distribution by Expertise:")
    for exp, role_levels in role_level_by_expertise_counts.items():
        print(f"\n  Expertise: {exp}")
        for role, resume_ids in role_levels.items():
            print(f"    {role}: {len(resume_ids)}")
            print(f"      Resumes: {', '.join(resume_ids)}")
    
    print("\nOrganizational Unit Distribution:")
    for unit, resume_ids in org_unit_counts.items():
        print(f"{unit}: {len(resume_ids)}")
        print(f"  Resumes: {', '.join(resume_ids)}")
    
    print("\nDetailed Resume Classifications:")
    for resume in processed_resumes:
        print(f"\nResume ID: {resume['resume_id']}")
        print(f"  Expertise: {', '.join(resume['expertise'])}")
        
        print("  Expertise by Job:")
        for job_id, job_expertise in resume['expertise_by_job'].items():
            print(f"    {job_id}: {', '.join(job_expertise)}")
        
        print("  Role Level by Expertise:")
        for exp, roles in resume['role_level_by_expertise'].items():
            print(f"    {exp}: {', '.join(roles)}")
            
        print(f"  Org Unit: {', '.join(resume['org_unit'])}")
    
    # Prepare for training
    datasets = cv_classifier.prepare_for_training(processed_resumes)
    
    # Train models (commented out as it would require GPU resources)
    '''cv_classifier.train_models(datasets)
    
    # Example of how to use for prediction (would need trained models)
    cv_classifier.load_trained_models()
    sample_resume_text = processed_resumes[0]['text']
    predictions = cv_classifier.predict(sample_resume_text)
    print("\nSample Predictions:")
    print(predictions)
    '''
    print("\nPipeline setup complete. To train models, uncomment the training section.")


if __name__ == "__main__":
    main() 