from abc import ABC, abstractmethod
from typing import Dict

class BasePrompt(ABC):
    def __init__(self, prompt: str):
        self.prompt = prompt

    @abstractmethod
    def set_prompt(self, **kwargs: Dict):
        pass

class UserPrompt(BasePrompt):
    def __init__(self, prompt: str):
        super().__init__(prompt)

    def set_prompt(self, **kwargs: Dict):
        return {"role": "user", 
                "content": self.prompt.format(**kwargs)}

class SystemPrompt(BasePrompt):
    def __init__(self, prompt: str):
        super().__init__(prompt)

    def set_prompt(self, **kwargs: Dict):
        return {"role": "system",
                "content": self.prompt.format(**kwargs)}

class AIResponse(BasePrompt):
    def __init__(self, prompt: str):
        super().__init__(prompt)

    def set_prompt(self, **kwargs: Dict):
        return {"role": "assitant",
                "content": self.prompt.format(**kwargs)}