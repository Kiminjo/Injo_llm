# IO
from abc import ABCMeta, abstractmethod
import numpy as np

# ETC 
from time import time

# Custom Libraries
from ..prompts import UserMessage, AIResponseMessage


class BaseTextModel(metaclass=ABCMeta):
    """
    The base class for all LLM models
    """
    def __init__(self):
        self.input_prompt = []

    @abstractmethod
    def load_llm(self, **kwargs):
        """
        Load the LLM model
        """
        pass

    def generate(self,
                 prompt: str,
                 save_previous: bool = True
                 ) -> str :
        """
        Generate a response from the model

        Args:
            - prompt: str
                The prompt for the model
            - save_previous: bool
                Whether to save the previous prompt

        Returns:
            - response: str
                The response from the model
        """

        if isinstance(prompt, str):
            prompt = UserMessage().set_prompt(prompt)

        # Set the prompt 
        self.input_prompt.append(prompt)

        # Generate the response
        start_time = time()
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=self.input_prompt,
        )
        latency = time() - start_time
        
        # Get the response and other information
        ai_message = response.choices[0].message.content
        self.latency = latency
        self.input_tokens = response.usage.prompt_tokens
        self.output_tokens = response.usage.completion_tokens
        self.total_tokens = response.usage.total_tokens

        # Save the previous prompt
        if save_previous:
            self.input_prompt.append(AIResponseMessage().set_prompt(ai_message))
        else: 
            self.input_prompt  = []
        
        return ai_message

        

class BaseTextEmbeddingModel(metaclass=ABCMeta):
    """
    The base class for all text embedding models
    """
    def __init__(self):
        pass

    def embedding(self, prompt: str | list[str]) -> list[float]:
        """
        Get the embedding from the prompt
        
        Args:
            - prompt: str, List[str]
                The prompt for the embedding
            
        Returns:
            - embedding_vector: List[float]
                The embedding vector from the model
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        # Get the embedding
        embedding_vector = self.llm_client.embeddings.create(
            model=self.model_name,
            input=prompt
        )
        embedding_vector = [vector.embedding for vector in embedding_vector.data]
        embedding_vector = np.array(embedding_vector)
        return embedding_vector