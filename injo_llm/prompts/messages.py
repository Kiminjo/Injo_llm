from abc import ABC, abstractmethod

class BaseMessage(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def set_prompt(self):
        pass 

class UserMessage(BaseMessage):
    def set_prompt(self, prompt: str):
        return {
            "role": "user",
            "content": prompt
        }
    
class SystemMessage(BaseMessage):
    def set_prompt(self, prompt: str):
        return {
            "role": "system",
            "content": prompt
        }


class AIResponseMessage(BaseMessage):
    def set_prompt(self, prompt: str):
        return {
            "role": "assistant",
            "content": prompt
        }
        
