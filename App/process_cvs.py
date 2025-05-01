import json
import os
from cv_agents.chains.classification_chain import CVClassificationChain
from tqdm import tqdm

def main():
    # Load CV data
    with open("silver_labeled_resumes.json", "r") as f:
        cv_data = json.load(f)
    
    # Initialize the chain
    chain = CVClassificationChain()
    
    # Process CVs
    results = []
    for cv in tqdm(cv_data[:10]):  # Start with a small batch
        try:
            result = chain.process_cv(cv)
            results.append(result)
            
            # Save intermediate results
            with open("agent_results.json", "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error processing CV {cv.get('resume_id', '')}: {e}")
    
    print(f"Processed {len(results)} CVs successfully")

if __name__ == "__main__":
    main()
