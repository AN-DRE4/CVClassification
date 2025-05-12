from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json
import time
import logging

class BaseAgent:
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, retry_delay=2):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.prompt = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def process(self, cv_data):
        """Process a CV with the agent with automatic retries"""
        if not self.prompt:
            raise NotImplementedError("Each agent must define its prompt")
        # Format the prompt with CV data
        formatted_prompt = self.prompt.format_prompt(**cv_data)
        # Initialize counters and tracking
        retries = 0
        result = None
        errors = []
        # Try processing with retries
        while retries <= self.max_retries:
            try:
                # Get response from LLM
                response = self.llm.invoke(formatted_prompt.to_messages())
                # Parse the response
                result = self._parse_response(response.content)
                # If we reach here, parsing was successful
                # For additional validation, check if the result is somewhat valid
                if self._validate_result(result):
                    return result
                else:
                    error_msg = f"Invalid result format after parsing: {result}"
                    logging.warning(error_msg)
                    errors.append(error_msg)
            except Exception as e:
                error_msg = f"Attempt {retries + 1} failed: {str(e)}"
                logging.warning(error_msg)
                errors.append(error_msg)
            
            # If we reach the max retries, break the loop
            if retries >= self.max_retries:
                break
                
            # Exponential backoff for retries
            sleep_time = self.retry_delay * (2 ** retries)
            logging.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            retries += 1
        
        # If we get here, all retries failed
        logging.error(f"All {self.max_retries} retry attempts failed")
        
        # Return a fallback result if we couldn't get a valid one
        return self._get_fallback_result(errors)
    
    def _parse_response(self, response_text):
        """Parse the LLM response into structured data"""
        raise NotImplementedError("Each agent must implement parsing logic")
    
    def _validate_result(self, result):
        """Validate the parsed result to ensure it's in the expected format
        Subclasses can override this for more specific validation."""
        return isinstance(result, dict) and len(result) > 0
    
    def _get_fallback_result(self, errors):
        """Provide a fallback result when all retries fail
        Subclasses should override this for specific fallback responses."""
        return {
            "error": True,
            "message": "Failed to process after multiple retries",
            "details": errors
        }
