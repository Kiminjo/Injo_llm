from abc import ABC, abstractmethod
from typing import Dict

class BasePromptTemplate(ABC):
    def __init__(self, prompt: str):
        self.prompt = prompt

    @abstractmethod
    def set_prompt(self, **kwargs: Dict):
        pass

class UserPromptTemplate(BasePromptTemplate):
    def __init__(self, prompt: str):
        super().__init__(prompt)

    def set_prompt(self, **kwargs: Dict):
        return {"role": "user", 
                "content": self.prompt.format(**kwargs)}

class SystemPromptTemplate(BasePromptTemplate):
    def __init__(self, prompt: str):
        super().__init__(prompt)

    def set_prompt(self, **kwargs: Dict):
        return {"role": "system",
                "content": self.prompt.format(**kwargs)}