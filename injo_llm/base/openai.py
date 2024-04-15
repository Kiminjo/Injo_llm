# Open AI Model
from openai import OpenAI

# IO
import os 
import sys 
import numpy as np
from pathlib import Path
from typing import List, Union, Dict

# Set working directory and system path
os.chdir(Path(__file__).parents[2])
sys.path.append(str(Path(__file__).parents[2]))

# Custom Libraries
from injo_llm.utils.prompt import fill_prompt

class BaseOpenAILLM:
    def __init__(self, api_key: str, chat_model: str = "gpt-3.5-turbo", embedding_model: str = "text-embedding-3-small"):
        # Define the OpenAI client
        self.llm_client = OpenAI(api_key=api_key)
        self.chat_model_name = chat_model 
        self.embedding_model_name = embedding_model

        # Define the message 
        self.input_prompts = []
        self._speaker = []

    def set_system_prompt(self, system_prompt: Union[str, List[str]], additional_info: Union[Dict, List[Dict]] = None):
        """
        Set the system prompt for the chat
        
        Args:
            - system_prompt: str or List[str]
                The system prompt for the chat
            - additional_info: Dict or List[Dict]
                The additional information to fill in the system prompt
        """
        if isinstance(system_prompt, str):
            system_prompt = [system_prompt]
        if isinstance(additional_info, Dict) or additional_info is None:
            additional_info = [additional_info]
        
        # Set the system prompt
        chat_system_prompt = []
        for prompt, info in zip(system_prompt, additional_info):
            if info is not None:
                prompt = fill_prompt(prompt, **info)
            sys_one_talk = {
                "role": "system",
                "content": prompt
            }
            chat_system_prompt.append(sys_one_talk)
        if "system" in self._speaker:
            user_idx = len(self._speaker) - self._speaker[::-1].index("system") - 1
            self.input_prompts = self.input_prompts[:user_idx] + chat_system_prompt + self.input_prompts[user_idx:]
            self._speaker = self._speaker[:user_idx] + ["system"] * len(chat_system_prompt) + self._speaker[user_idx:]
        else:
            self.input_prompts = chat_system_prompt + self.input_prompts
            self._speaker = ["system"] * len(chat_system_prompt) + self._speaker

    def set_human_prompt(self, human_prompt: Union[str, List[str]], additional_info: Union[Dict, List[Dict]] = None):
        """
        Set the human prompt for the chat
        
        Args:
            - human_prompt: str or List[str]
                The human prompt for the chat
            - additional_info: Dict or List[Dict]
                The additional information to fill in the human prompt
        """
        if isinstance(human_prompt, str):
            human_prompt = [human_prompt]
        if isinstance(additional_info, Dict) or additional_info is None:
            additional_info = [additional_info]

        # Set the human prompt
        chat_human_prompt = []
        for prompt, info in zip(human_prompt, additional_info):
            if info is not None:
                prompt = fill_prompt(prompt, **info)
            human_one_talk = {
                "role": "user",
                "content": prompt
            }
            chat_human_prompt.append(human_one_talk)
        
        if "system" in self._speaker:
            system_idx = len(self._speaker) - self._speaker[::-1].index("system") - 1
            self.input_prompts = self.input_prompts[:system_idx] + chat_human_prompt + self.input_prompts[system_idx:]
            self._speaker = self._speaker[:system_idx] + ["user"] * len(chat_human_prompt) + self._speaker[system_idx:]
        else:
            self.input_prompts = chat_human_prompt + self.input_prompts
            self._speaker = ["user"] * len(chat_human_prompt) + self._speaker

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

    def generate(self, prompt: str, additional_info: Dict = None):
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
        # Make full prompt 
        if additional_info is not None:
            prompt = fill_prompt(prompt, **additional_info)

        # Set the user prompt 
        prompt = {
            "role": "user",
            "content": prompt
        }
        self.input_prompts.append(prompt)
        self._speaker.append("user")

        # Generate the answer 
        answer = self.llm_client.chat.completions.create(
            model=self.chat_model_name,
            messages=self.input_prompts
        )
        answer = answer.choices[0].message.content

        # Save the answer to the chat history 
        self.input_prompts.append({
            "role": "assistant",
            "content": answer
        })
        self._speaker.append("assistant")
        return answer