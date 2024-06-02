# IO
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union

# ETC 
from time import time 

# Custom Libraries
from src.text.prompts import BasePrompt, UserPrompt, AIResponse

class BaseLLM(metaclass=ABCMeta):
    """
    The base class for all LLM models
    """
    def __init__(self,
                 base_prompt: Union[List[BasePromptTemplate], BasePromptTemplate] = None
                 ):
        if isinstance(base_prompt, str):
            raise ValueError("base_prompt should be a list of BasePromptTemplate objects")
        
        if isinstance(base_prompt, BasePromptTemplate):
            base_prompt = [base_prompt]

        self.input_messages = []
        if base_prompt is not None:
            self.input_messages += base_prompt

    @abstractmethod
    def load_llm(self, **kwargs):
        """
        Load the LLM model
        """
        pass

    def generate(self, 
                 prompt: BasePromptTemplate, 
                 additional_info: Dict = {}):
        """
        Generate the answer from the prompt
        
        Args:
            - prompt: str
                The prompt for the chat
            - additional_info: Dict
                The additional information to fill in the prompt
        
        Returns:
            - answer: str
                The answer from the model
        """

        # Set the user prompt 
        user_prompt = UserMessage().set_prompt(prompt)
        self.input_messages.append(user_prompt)

        # Generate the answer 
        # Measure the time it takes to generate the answer
        start_time = time()
        
        answer = self.llm_client.chat.completions.create(
            model=self.chat_model_name,
            messages=self.input_messages
        )

        # Save the latency and tokens 
        self.latency = time() - start_time
        self.input_tokens = answer.usage.prompt_tokens
        self.output_tokens = answer.usage.completion_tokens
        self.total_tokens = answer.usage.total_tokens

        # Get the answer as string form 
        answer = answer.choices[0].message.content

        # Save the answer to the chat history 
        ai_response = AIResponseMessage().set_prompt(answer)
        self.input_messages.append(ai_response)

        return answer
    
    def clear(self, mode: str = "last"):
        """
        Clear the chat history
        """
        if mode == "all":
            self.input_messages = []
        
        elif mode == "last":
            self.input_messages = self.input_messages[:-2]
        else:
            raise ValueError("mode should be either 'all' or 'last'")