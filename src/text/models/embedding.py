# Open AI Model
from openai import OpenAI

# Custom Libraries
from .base import BaseTextEmbeddingModel

class OpenAITextEmbedding(BaseTextEmbeddingModel):
    def __init__(self, 
                 api_key: str, 
                 model_name: str = "text-embedding-ada-002", 
                 ):
        super().__init__()

        # Define the OpenAI client
        self.llm_client = self.load_llm(api_key=api_key)
        self.model_name = model_name

    def load_llm(self, api_key: str) -> OpenAI:
        return OpenAI(api_key=api_key)

class OllamaTextEmbedding(BaseTextEmbeddingModel):
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
                       api_key="ollama")    

class LMStudioTextEmbedding(BaseTextEmbeddingModel):
    def __init__(self,
                 base_url: str = "http://localhost:1234/v1",
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
                       api_key="lmstudio")
    
