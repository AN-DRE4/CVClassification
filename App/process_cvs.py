import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import argparse
from cv_agents.chains.classification_chain import CVClassificationOrchestrator

class CVProcessor:
    def __init__(self, input_file: str, output_dir: str = "results", custom_config: Optional[Dict[str, Any]] = None, 
                 config_files: Optional[List[str]] = None, 
                 interpreter_configs: Optional[List[Tuple[str, str, str]]] = None,
                 model_name: str = "gpt-4o-mini-2024-07-18"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_name = model_name
        
        # Initialize orchestrator with base custom configuration and model
        self.orchestrator = CVClassificationOrchestrator(custom_config=custom_config, model_name=model_name)
        
        # Load additional configuration from files if provided
        if config_files:
            for config_file in config_files:
                self.load_config_from_file(config_file)
        
        # Load and apply interpreter configurations if provided
        if interpreter_configs:
            for file_path, description, agent_type in interpreter_configs:
                self.load_config_from_interpreter(file_path, description, agent_type)
                
        self.results: List[Dict] = []
        self.errors: List[Dict] = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_config_from_file(self, config_file: str) -> bool:
        """Load configuration from a file and apply it to the orchestrator"""
        try:
            return self.orchestrator.load_config_from_file(config_file)
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            return False
    
    def load_config_from_interpreter(self, file_path: str, interpretation_description: str, agent_type: str) -> bool:
        """Load configuration using the interpreter agent"""
        try:
            return self.orchestrator.load_config_from_interpreter(file_path, interpretation_description, agent_type)
        except Exception as e:
            print(f"Error loading configuration using interpreter: {e}")
            return False
    
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
            expertises = result.get("expertise", {}).get("expertise", [])
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
                    print(f"Error processing CV in processing_cvs.py: {e}")
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

        return self.results

def main():
    print("Starting CV processing...")
    parser = argparse.ArgumentParser(description='Process CVs with classification agents')
    parser.add_argument('--input', type=str, default='silver_labeled_resumes.json', help='Path to the input JSON file containing CV data')
    parser.add_argument('--output', type=str, default='agents_results', help='Path to the output directory for results')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of CVs to process in each batch')
    parser.add_argument('--save_interval', type=int, default=5, help='Save results every N batches')
    parser.add_argument('--max_cvs', type=int, default=None, help='Maximum number of CVs to process, if None, all CVs will be processed')
    parser.add_argument('--clear_memory', type=bool, default=False, help='Clear memory before processing')
    parser.add_argument('--config', type=str, action='append', help='Path to configuration file(s) for customizing agent behavior')
    parser.add_argument('--model', type=str, default='gpt-4o-mini-2024-07-18', help='OpenAI model to use for classification')
    
    # Add interpreter configuration arguments
    parser.add_argument('--expertise-file', type=str, help='Path to file containing expertise categories information')
    parser.add_argument('--expertise-description', type=str, help='Description of how to interpret the expertise file')
    parser.add_argument('--role-file', type=str, help='Path to file containing role levels information')
    parser.add_argument('--role-description', type=str, help='Description of how to interpret the role file')
    parser.add_argument('--org-file', type=str, help='Path to file containing organizational units information')
    parser.add_argument('--org-description', type=str, help='Description of how to interpret the org file')
    
    args = parser.parse_args()
    print(f"Processing CVs from {args.input} to {args.output}")
    
    # Configuration
    input_file = args.input if args.input else 'silver_labeled_resumes.json'
    output_dir = args.output if args.output else 'agents_results'
    batch_size = args.batch_size if args.batch_size else 10
    save_interval = args.save_interval if args.save_interval else 5
    max_cvs = args.max_cvs if args.max_cvs else None
    clear_memory = args.clear_memory if args.clear_memory else False
    config_files = args.config if args.config else []
    model_name = args.model if args.model else "gpt-4o-mini-2024-07-18"
    
    # Process interpreter configurations
    interpreter_configs = []
    
    if args.expertise_file and args.expertise_description:
        interpreter_configs.append((args.expertise_file, args.expertise_description, "expertise"))
    
    if args.role_file and args.role_description:
        interpreter_configs.append((args.role_file, args.role_description, "role_levels"))
    
    if args.org_file and args.org_description:
        interpreter_configs.append((args.org_file, args.org_description, "org_units"))
    
    # Initialize and run processor
    processor = CVProcessor(
        input_file=input_file, 
        output_dir=output_dir, 
        config_files=config_files,
        interpreter_configs=interpreter_configs,
        model_name=model_name
    )
    
    if clear_memory:
        print("Clearing memory...")
        processor.orchestrator.clear_memory()
    
    print("Processing CVs...")
    processor.process_cvs(batch_size, save_interval, max_cvs)

if __name__ == "__main__":
    main()
