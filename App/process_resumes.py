from active_learning_pipeline import ActiveLearningPipeline

def main():
    # Initialize pipeline
    pipeline = ActiveLearningPipeline()
    
    # Load data (only unlabeled resumes needed)
    pipeline.load_data(
        labeled_file='silver_labeled_resumes.json',  # Needed for label binarizer
        unlabeled_dir='CVs'
    )
    
    # Load trained model (won't retrain)
    training_dataset = pipeline.prepare_training_data()
    pipeline.train_model(training_dataset, force_retrain=False)
    
    # Process all remaining resumes
    processed_resumes = pipeline.process_all_resumes()

if __name__ == "__main__":
    main() 