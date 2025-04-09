import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
import os
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import logging
from human_labeling_interface import HumanLabelingInterface
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActiveLearningPipeline:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_binarizer = MultiLabelBinarizer()
        self.training_data = []
        self.unlabeled_data = []
        self.batch_size = 5  # Number of samples to select for labeling in each iteration
        self.labeling_interface = HumanLabelingInterface()
        
    def load_data(self, labeled_file: str, unlabeled_dir: str):
        """Load labeled and unlabeled data"""
        # Load labeled data
        with open(labeled_file, 'r') as f:
            self.training_data = json.load(f)
            logger.info(f"Loaded {len(self.training_data)} labeled resumes")
        
        # Load unlabeled data
        self.unlabeled_data = []
        for root, _, files in os.walk(unlabeled_dir):
            for file in files:
                if file.endswith('.txt'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        self.unlabeled_data.append({
                            'id': os.path.join(root, file),
                            'text': f.read()
                        })
        logger.info(f"Loaded {len(self.unlabeled_data)} unlabeled resumes")
    
    def prepare_training_data(self) -> Dataset:
        """Prepare training data for model training"""
        # Extract text and labels
        texts = []
        labels = []
        
        for resume in self.training_data:
            texts.append(resume['resume_text'])
            
            # Extract expertise categories from work experience and extra skills
            expertise = set()
            
            # Extract from work experience
            if 'extracted_info' in resume and 'work_experience' in resume['extracted_info']:
                for job in resume['extracted_info']['work_experience']:
                    if isinstance(job, dict):
                        # Extract from technical skills
                        if 'technical_skills' in job and isinstance(job['technical_skills'], list):
                            for skill in job['technical_skills']:
                                if isinstance(skill, str):
                                    expertise.update(self._map_skill_to_expertise(skill))
                        
                        # Extract from responsibilities
                        if 'responsibilities' in job and isinstance(job['responsibilities'], list):
                            for resp in job['responsibilities']:
                                if isinstance(resp, str):
                                    expertise.update(self._map_skill_to_expertise(resp))
            
            # Extract from extra skills
            if 'extracted_info' in resume and 'extra_skills' in resume['extracted_info']:
                extra_skills = resume['extracted_info']['extra_skills']
                if isinstance(extra_skills, dict) and 'Technical skills' in extra_skills:
                    for skill in extra_skills['Technical skills']:
                        if isinstance(skill, str):
                            expertise.update(self._map_skill_to_expertise(skill))
            
            labels.append(list(expertise) if expertise else ['unknown'])
        
        # Fit and transform labels
        label_matrix = self.label_binarizer.fit_transform(labels)
        
        # Create dataset
        dataset = Dataset.from_dict({
            'text': texts,
            'label': label_matrix.astype(np.float32).tolist()
        })
        
        return dataset
    
    def _map_skill_to_expertise(self, skill: str) -> List[str]:
        """Map a skill to corresponding expertise categories"""
        skill_lower = skill.lower()
        expertise = []
        
        # Define expertise categories and their keywords
        expertise_categories = {
            'software_development': ['python', 'java', 'c#', 'javascript', 'html', 'css', 'react', 'angular', 'vue', 'node.js'],
            'data_engineering': ['sql', 'etl', 'hadoop', 'spark', 'aws', 'data warehouse', 'azure', 'databricks'],
            'data_science': ['machine learning', 'deep learning', 'ai', 'artificial intelligence', 'statistics', 'tensorflow', 'pytorch'],
            'devops': ['jenkins', 'docker', 'kubernetes', 'ci/cd', 'terraform', 'ansible', 'linux'],
            'cybersecurity': ['security', 'penetration testing', 'network security', 'firewall', 'encryption'],
            'marketing': ['seo', 'content writing', 'google analytics', 'social media', 'digital marketing'],
            'finance': ['excel', 'accounting', 'financial analysis', 'banking', 'reconciliation'],
            'management': ['project management', 'team lead', 'leadership', 'agile', 'scrum']
        }
        
        for category, keywords in expertise_categories.items():
            if any(keyword.lower() in skill_lower for keyword in keywords):
                expertise.append(category)
        
        return expertise
    
    def train_model(self, dataset: Dataset, force_retrain: bool = False):
        """Train the classification model or load existing one"""
        model_dir = "./cv_classifier"
        binarizer_path = os.path.join(model_dir, "label_binarizer.pkl")
        
        # Check if model exists and we don't want to force retrain
        if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")) and not force_retrain:
            logger.info("Loading existing model...")
            try:
                # Load label binarizer
                with open(binarizer_path, 'rb') as f:
                    self.label_binarizer = pickle.load(f)
                
                # Load model and tokenizer
                self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                logger.info("Successfully loaded existing model and label binarizer")
                return
            except Exception as e:
                logger.warning(f"Error loading existing model: {e}")
                logger.warning("Will train a new model instead")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Split dataset
        dataset_dict = dataset.train_test_split(test_size=0.2)
        
        # Tokenize datasets
        tokenized_datasets = dataset_dict.map(
            lambda x: self.tokenizer(x['text'], padding='max_length', truncation=True, max_length=512),
            batched=True
        )
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_binarizer.classes_),
            problem_type="multi_label_classification"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=model_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
        )
        
        try:
            # Train model
            trainer.train()
            
            # Save model, tokenizer, and label binarizer
            self.model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)
            with open(binarizer_path, 'wb') as f:
                pickle.dump(self.label_binarizer, f)
            
            logger.info("Model training completed and saved successfully")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def select_samples_for_labeling(self) -> List[Dict[str, Any]]:
        """Select samples for labeling using uncertainty sampling"""
        if not self.model or not self.unlabeled_data:
            return []
        
        # Process in smaller batches to avoid memory issues
        batch_size = 100  # Process 100 resumes at a time
        all_uncertainties = []
        all_indices = []
        
        for i in range(0, len(self.unlabeled_data), batch_size):
            batch = self.unlabeled_data[i:i + batch_size]
            texts = [resume['text'] for resume in batch]
            
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(self.unlabeled_data) + batch_size - 1)//batch_size}")
            logger.debug(f"Processing {len(batch)} resumes from index {i} to {i + len(batch)}")
            
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.sigmoid(outputs.logits)
            
            # Calculate uncertainty (entropy)
            uncertainty = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
            
            # Store uncertainties and indices
            all_uncertainties.extend(uncertainty.tolist())
            all_indices.extend(range(i, i + len(batch)))
            
            logger.debug(f"Batch uncertainty scores - Min: {uncertainty.min():.4f}, Max: {uncertainty.max():.4f}, Mean: {uncertainty.mean():.4f}")
        
        # Convert to numpy arrays for easier manipulation
        uncertainties = np.array(all_uncertainties)
        indices = np.array(all_indices)
        
        # Select top k most uncertain samples
        top_k = min(self.batch_size, len(self.unlabeled_data))
        top_k_indices = indices[np.argsort(uncertainties)[-top_k:]]
        
        selected_samples = [self.unlabeled_data[i] for i in top_k_indices]
        
        # Remove selected samples from unlabeled data
        self.unlabeled_data = [resume for i, resume in enumerate(self.unlabeled_data) if i not in top_k_indices]
        
        return selected_samples
    
    def add_labeled_samples(self, labeled_samples: List[Dict[str, Any]]):
        """Add newly labeled samples to training data"""
        self.training_data.extend(labeled_samples)
        logger.info(f"Added {len(labeled_samples)} new labeled samples")
    
    def save_labeled_data(self, output_file: str):
        """Save labeled data to file"""
        with open(output_file, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        logger.info(f"Saved labeled data to {output_file}")

    def process_all_resumes(self) -> List[Dict[str, Any]]:
        """Process all remaining unlabeled resumes using the trained model"""
        if not self.model:
            logger.error("No trained model available. Please train the model first.")
            return []
        
        logger.info(f"Processing {len(self.unlabeled_data)} remaining resumes...")
        processed_resumes = []
        batch_size = 100  # Process 100 resumes at a time
        
        for i in range(0, len(self.unlabeled_data), batch_size):
            batch = self.unlabeled_data[i:i + batch_size]
            texts = [resume['text'] for resume in batch]
            
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(self.unlabeled_data) + batch_size - 1)//batch_size}")
            
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.sigmoid(outputs.logits)
            
            # Convert probabilities to predictions
            predictions = (probabilities > 0.5).float().numpy()
            
            # Map predictions back to expertise categories
            for j, resume in enumerate(batch):
                expertise_indices = np.where(predictions[j] == 1)[0]
                expertise_categories = [self.label_binarizer.classes_[idx] for idx in expertise_indices]
                
                # Create extracted info structure
                extracted_info = {
                    'work_experience': [{
                        'technical_skills': [],
                        'soft_skills': []
                    }],
                    'extra_skills': {
                        'Technical skills': [],
                        'Soft skills': []
                    }
                }
                
                # Add predicted expertise categories to skills
                for expertise in expertise_categories:
                    if expertise != 'unknown':
                        extracted_info['extra_skills']['Technical skills'].append(expertise)
                
                processed_resumes.append({
                    'resume_id': resume['id'],
                    'resume_text': resume['text'],
                    'extracted_info': extracted_info,
                    'predicted_expertise': expertise_categories
                })
        
        # Save processed resumes
        output_file = 'processed_resumes.json'
        with open(output_file, 'w') as f:
            json.dump(processed_resumes, f, indent=2)
        
        logger.info(f"Processed {len(processed_resumes)} resumes. Results saved to {output_file}")
        return processed_resumes

def main():
    # Initialize pipeline
    pipeline = ActiveLearningPipeline()
    
    # Load data
    pipeline.load_data(
        labeled_file='silver_labeled_resumes.json',
        unlabeled_dir='CVs'
    )
    
    # Initial training
    logger.info("Starting initial training...")
    training_dataset = pipeline.prepare_training_data()
    pipeline.train_model(training_dataset, force_retrain=False)  # Will load existing model if available
    
    # Active learning loop
    num_iterations = 5  # Number of active learning iterations
    for iteration in range(num_iterations):
        logger.info(f"\nStarting iteration {iteration + 1}/{num_iterations}")
        
        # Select samples for labeling
        selected_samples = pipeline.select_samples_for_labeling()
        
        logger.info(f"Done selecting samples for labeling")
        
        if not selected_samples:
            logger.info("No more unlabeled samples available")
            break
        
        # Save samples for human labeling
        labeling_file = pipeline.labeling_interface.save_samples_for_labeling(selected_samples, iteration + 1)
        
        # Print instructions for human labelers
        logger.info("\n" + pipeline.labeling_interface.get_labeling_instructions())
        logger.info(f"\nPlease label the samples in: {labeling_file}")
        logger.info("After labeling, press Enter to continue...")
        input()
        
        # Load labeled samples
        labeled_samples = pipeline.labeling_interface.load_labeled_samples(labeling_file)
        
        if labeled_samples:
            # Add newly labeled samples to training data
            pipeline.add_labeled_samples(labeled_samples)
            
            # Retrain model with expanded dataset
            logger.info("Retraining model with expanded dataset...")
            training_dataset = pipeline.prepare_training_data()
            pipeline.train_model(training_dataset)
            
            # Save current state
            pipeline.save_labeled_data(f'silver_labeled_resumes_iteration_{iteration + 1}.json')
        else:
            logger.warning("No valid labeled samples found. Skipping retraining.")
        
        logger.info(f"Completed iteration {iteration + 1}")

    # After active learning loop completes, process remaining resumes
    logger.info("\nProcessing all remaining resumes...")
    processed_resumes = pipeline.process_all_resumes()
    logger.info("Pipeline completed!")

if __name__ == "__main__":
    main() 