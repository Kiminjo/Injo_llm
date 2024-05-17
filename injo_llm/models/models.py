# Open AI Model
from openai import OpenAI
from openai.lib.azure import AzureOpenAI
from groq import Groq

# IO
import numpy as np
from typing import List, Union

# Custom Libraries
from .base import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self, 
                 api_key: str, 
                 chat_model: str = "gpt-3.5-turbo", 
                 embedding_model: str = "text-embedding-3-small"
                 ):
        super().__init__()

        # Define the OpenAI client
        self.llm_client = self.load_llm(api_key=api_key)
        self.chat_model_name = chat_model 
        self.embedding_model_name = embedding_model

    def load_llm(self, api_key: str) -> OpenAI:
        return OpenAI(api_key=api_key)
    
    def embedding(self, texts: Union[str, List[str]]) -> List[float]:
        """
        Get the embedding from the prompt
        
        Args:
            - texts: str, List[str]
                The prompt for the embedding
            
        Returns:
            - embedding_vector: List[float]
                The embedding vector from the model
        """
        if isinstance(texts, str):
            texts = [texts]

        # Get the embedding
        embedding_vector = self.llm_client.embeddings.create(
            model=self.embedding_model_name,
            input=texts
        )
        embedding_vector = [vector.embedding for vector in embedding_vector.data]
        embedding_vector = np.array(embedding_vector)
        return embedding_vector
    

class AzureLLM(BaseLLM):
    def __init__(self, 
                 api_key: str, 
                 chat_model: str = "gpt-3.5-turbo-1106",
                 api_version: str = "2023-12-01-preview",
                 endpoint: str = "https://api.openai.com",
                ):
        super().__init__()
        
        # Define the OpenAI client
        self.llm_client = self.load_llm(api_key=api_key,
                                        api_version=api_version,
                                        endpoint=endpoint)
        self.chat_model_name = chat_model 

    def load_llm(self, 
                 api_key: str,
                 api_version: str,
                 endpoint: str
                 ) -> AzureOpenAI:
        return AzureOpenAI(api_key=api_key,
                           api_version=api_version,
                           azure_endpoint=endpoint)
    
class Llama3LLM(BaseLLM):
    def __init__(self, 
                 api_key: str, 
                 chat_model: str = "llama3-70b-8192"
                ):
        super().__init__()
        
        # Define the OpenAI client
        self.llm_client = self.load_llm(api_key=api_key)
        self.chat_model_name = chat_model 

    def load_llm(self, 
                 api_key: str,
                 ):
        return Groq(api_key=api_key)