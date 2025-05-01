import json
import os
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
from cv_agents.chains.classification_chain import CVClassificationOrchestrator

class CVProcessor:
    def __init__(self, input_file: str, output_dir: str = "results"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.orchestrator = CVClassificationOrchestrator()
        self.results: List[Dict] = []
        self.errors: List[Dict] = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_cv_data(self) -> List[Dict]:
        """Load CV data from input file"""
        try:
            with open(self.input_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading input file: {e}")
            return []
    
    def save_results(self, batch: bool = False):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_results_{timestamp}.json" if batch else "results.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "total_processed": len(self.results),
                "total_errors": len(self.errors),
                "results": self.results,
                "errors": self.errors
            }, f, indent=2)
    
    def analyze_results(self):
        """Analyze and print statistics about the results"""
        print("\nResults Analysis:")
        print(f"Total CVs processed: {len(self.results)}")
        print(f"Total errors: {len(self.errors)}")
        
        # Analyze sources
        sources = {}
        for result in self.results:
            source = result.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        print("\nClassification Sources:")
        for source, count in sources.items():
            print(f"- {source}: {count}")
        
        # Analyze expertise areas
        expertise_areas = {}
        for result in self.results:
            expertises = result.get("expertise", [])['expertise']
            for exp in expertises:
                category = exp.get("category")
                expertise_areas[category] = expertise_areas.get(category, 0) + 1
        
        print("\nTop Expertise Areas:")
        for category, count in sorted(expertise_areas.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"- {category}: {count}")
    
    def process_cvs(self, batch_size: int = 10, save_interval: int = 5, max_cvs: int = 100):
        """Process CVs in batches with progress tracking"""
        cv_data = self.load_cv_data()
        if not cv_data:
            return
        
        if max_cvs is not None:
            print(f"Processing {max_cvs} CVs...")
        else:
            max_cvs = len(cv_data)
            print(f"Processing {max_cvs} CVs...")
        
        # Process in batches
        for i in range(0, len(cv_data), batch_size):
            if i >= max_cvs:
                break
            batch = cv_data[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1} of {len(cv_data)//batch_size + 1}")
            
            for cv in tqdm(batch, desc="Processing CVs"):
                try:
                    result = self.orchestrator.process_cv(cv)
                    self.results.append(result)
                except Exception as e:
                    self.errors.append({
                        "resume_id": cv.get("resume_id", "unknown"),
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Save intermediate results
            if (i + batch_size) % save_interval == 0:
                self.save_results(batch=True)
                print(f"Saved intermediate results after {i + batch_size} CVs")
        
        # Save final results
        self.save_results()
        self.analyze_results()

def main():
    # Configuration
    input_file = "silver_labeled_resumes.json"
    output_dir = "agents_results"
    batch_size = 10
    save_interval = 5
    max_cvs = 10 # Number of CVs to process, if you want to process all, set to None
    
    # Initialize and run processor
    processor = CVProcessor(input_file, output_dir)
    processor.process_cvs(batch_size, save_interval, max_cvs)

if __name__ == "__main__":
    main()
