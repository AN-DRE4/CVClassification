from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json

class BaseAgent:
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.prompt = None
        
    def process(self, cv_data):
        """Process a CV with the agent"""
        if not self.prompt:
            raise NotImplementedError("Each agent must define its prompt")
        
        # Format the prompt with CV data
        formatted_prompt = self.prompt.format_prompt(**cv_data)
        
        # Get response from LLM
        response = self.llm.invoke(formatted_prompt.to_messages())
        
        # Parse and return structured output
        return self._parse_response(response.content)
    
    def _parse_response(self, response_text):
        """Parse the LLM response into structured data"""
        raise NotImplementedError("Each agent must implement parsing logic")
