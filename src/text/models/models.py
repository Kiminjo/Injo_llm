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
    
    

class GroqLLM(BaseLLM):
    def __init__(self, 
                 api_key: str, 
                 chat_model: str = "llama3-70b-8192"
                ):
        super().__init__()
        
        # Define the Groq client
        self.llm_client = self.load_llm(api_key=api_key)
        self.chat_model_name = chat_model 

    def load_llm(self, 
                 api_key: str,
                 ):
        return Groq(api_key=api_key)
    
class OllamaLLM(BaseLLM):
    def __init__(self,
                 api_key: str,
                 base_url: str = "http://localhost:11434/v1",
                 chat_model: str = "llama3:latest",
                 embedding_model: str = "llama3:latest"
                 ):
        super().__init__()

        # Define the LM Studio client
        self.llm_client = self.load_llm(base_url=base_url, api_key=api_key)
        self.chat_model_name = chat_model
        self.embedding_model_name = embedding_model

    def load_llm(self, 
                 base_url: str, 
                 api_key: str
                 ) -> OpenAI:
         return OpenAI(base_url=base_url,
                       api_key=api_key)
    
    def query(self, texts: Union[str, List[str]]) -> List[float]:
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
    

class LMStudioLLM(BaseLLM):
    def __init__(self,
                 api_key: str,
                 base_url: str = "http://localhost:1234/v1",
                 chat_model: str = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
                 embedding_model: str = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf"
                 ):
        super().__init__()

        # Define the LM Studio client
        self.llm_client = self.load_llm(base_url=base_url, api_key=api_key)
        self.chat_model_name = chat_model
        self.embedding_model_name = embedding_model

    def load_llm(self, 
                 base_url: str, 
                 api_key: str
                 ) -> OpenAI:
         return OpenAI(base_url=base_url,
                       api_key=api_key)
    
    def query(self, texts: Union[str, List[str]]) -> List[float]:
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