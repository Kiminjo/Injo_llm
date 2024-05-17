# IO
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union

# Custom Libraries
from injo_llm.prompts.prompts import BasePrompt, UserPrompt, AIResponse

class BaseLLM(metaclass=ABCMeta):
    """
    The base class for all LLM models
    """
    def __init__(self,
                 base_prompt: Union[List[BasePrompt], BasePrompt] = None
                 ):
        if isinstance(base_prompt, str):
            raise ValueError("base_prompt should be a list of BasePrompt objects")
        
        if isinstance(base_prompt, BasePrompt):
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
                 prompt: BasePrompt, 
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
        user_template = UserPrompt(prompt=prompt)
        user_prompt = user_template.set_prompt(**additional_info)
        self.input_messages.append(user_prompt)

        # Generate the answer 
        answer = self.llm_client.chat.completions.create(
            model=self.chat_model_name,
            messages=self.input_messages
        )
        answer = answer.choices[0].message.content

        # Save the answer to the chat history 
        ai_template = AIResponse(prompt=answer)
        ai_response = ai_template.set_prompt()
        self.input_messages.append(ai_response)

        return answer
    