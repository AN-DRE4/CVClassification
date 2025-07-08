from ..base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
import json
import logging
import os
from typing import Dict, Any, Optional
import re
import pandas as pd

INTERPRETER_SYSTEM_PROMPT = """You are an expert Interpreter Agent specializing in understanding structured data files based on user descriptions.

Your task is to analyze the provided file content and the user's description of how to interpret it.
You should extract structured information that can be used to guide CV classification.

The output should be a well-structured configuration that follows these guidelines:
1. For expertise categories: extract a list of expertise categories with relevant metadata
2. For role levels: extract role level definitions with clear descriptions
3. For organizational units: extract org unit definitions with clear descriptions

Provide your analysis in a structured JSON format appropriate for the agent type.
DO NOT include any explanations or text outside the JSON structure.
"""

INTERPRETER_USER_PROMPT = """I need you to interpret the following file for a {agent_type} agent.

Description of how to interpret this file:
{interpretation_description}

File content:
{file_content}

Your task is to convert this information into a structured format that the {agent_type} agent can use for CV classification.
For an expertise agent, format your response as a valid JSON object with "expertise_categories" as the key containing an array of objects, 
each being a string of the expertise category name.
For a role level agent, format your response as a valid JSON object with "role_levels" as the key containing an array of objects, 
each being an object with "name" and "description" fields.
For an org unit agent, format your response as a valid JSON object with "org_units" as the key containing an array of objects, 
each being an object with "name" and "description" fields.

Return ONLY the properly formatted JSON configuration.
"""

class InterpreterAgent(BaseAgent):
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, retry_delay=2):
        super().__init__(model_name, temperature, max_retries, retry_delay)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", INTERPRETER_SYSTEM_PROMPT),
            ("human", INTERPRETER_USER_PROMPT)
        ])
    
    def _parse_response(self, response_text):
        """Parse the LLM JSON response"""
        try:
            # Clean and parse the JSON response
            # Remove any markdown formatting if present
            response_text = clean_json_string(response_text)                
            response_text = response_text.strip()
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {response_text}")
            logging.error(f"JSON error: {str(e)}")
            raise ValueError("Response is not valid JSON")
    
    def _validate_result(self, result):
        """Validate the interpreter result structure"""
        if not isinstance(result, dict):
            return False
        
        # Basic validation - ensure it's a non-empty dictionary
        return len(result) > 0
    
    def _get_fallback_result(self, errors):
        """Generate fallback interpreter result"""
        logging.warning(f"Using fallback result for interpreter agent after errors: {errors}")
        return {
            "error": True,
            "message": "Failed to interpret the provided file",
            "details": errors
        }
    
    def interpret_file(self, file_content: str, interpretation_description: str, agent_type: str) -> Dict[str, Any]:
        """Interpret a file based on the provided description and agent type"""
        input_data = {
            "file_content": file_content,
            "interpretation_description": interpretation_description,
            "agent_type": agent_type
        }
        
        return self.process(input_data)
    
    @staticmethod
    def process_file_for_agent(file_path: str, interpretation_description: str, agent_type: str) -> Optional[Dict[str, Any]]:
        """Process a file and return a configuration for the specified agent type"""
        try:
            # Read the file content
            if file_path.endswith('.xlsx'):
                file_content = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                file_content = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    file_content = json.load(f)
            elif file_path.endswith('.txt'):
                with open(file_path, 'r') as f:
                    file_content = f.read()
            else:
                with open(file_path, 'r') as f:
                    file_content = f.read()
            
            # Create an interpreter agent and process the file
            interpreter = InterpreterAgent()
            result = interpreter.interpret_file(file_content, interpretation_description, agent_type)
            
            if "error" in result and result["error"]:
                logging.error(f"Error interpreting file: {result.get('message', 'Unknown error')}")
                return None
                
            return result
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return None 
        
def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()