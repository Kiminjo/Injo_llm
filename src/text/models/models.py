# Open AI Model
from openai import OpenAI
from groq import Groq

# Custom Libraries
from .base import BaseTextModel

class OpenAITextLLM(BaseTextModel):
    def __init__(self, 
                 api_key: str, 
                 model_name: str = "gpt-3.5-turbo", 
                 ):
        super().__init__()

        # Define the OpenAI client
        self.llm_client = self.load_llm(api_key=api_key)
        self.model_name = model_name

    def load_llm(self, api_key: str) -> OpenAI:
        return OpenAI(api_key=api_key)
    

class GroqTextLLM(BaseTextModel):
    def __init__(self, 
                 api_key: str, 
                 model_name: str = "llama3-70b-8192"
                ):
        super().__init__()
        
        # Define the Groq client
        self.llm_client = self.load_llm(api_key=api_key)
        self.model_name = model_name 

    def load_llm(self, 
                 api_key: str,
                 ):
        return Groq(api_key=api_key)
    

class OllamaTextLLM(BaseTextModel):
    def __init__(self,
                 base_url: str = "http://localhost:11434/v1",
                 model_name: str = "llama3:latest",
                 ):
        super().__init__()

        # Define the LM Studio client
        self.llm_client = self.load_llm(base_url=base_url)
        self.model_name = model_name

    def load_llm(self, 
                 base_url: str, 
                 ) -> OpenAI:
         return OpenAI(base_url=base_url,
                       api_key="ollma")
    

class LMStudioTextLLM(BaseTextModel):
    def __init__(self,
                 base_url: str = "http://localhost:1234/v1",
                 model_name: str = "llama3:latest",
                 ):
        
        super().__init__()

        # Define the LM Studio client
        self.llm_client = self.load_llm(base_url=base_url)
        self.model_name = model_name

    def load_llm(self, 
                 base_url: str
                 ) -> OpenAI:
         return OpenAI(base_url=base_url,
                       api_key="lmstudio")
    