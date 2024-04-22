# LLM Model
from openai.lib.azure import AzureOpenAI

# Custom Libraries
from .base import BaseLLM

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