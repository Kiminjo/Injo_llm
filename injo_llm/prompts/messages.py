class UserMessage:
    def __init__(self, prompt: str):
        self.prompt = {
            "role": "user",
            "content": prompt
        }
    
class SystemMessage:
    def __init__(self, prompt: str):
        self.prompt = {
            "role": "system",
            "content": prompt
        }


class AIResponseMessage:
    def __init__(self, prompt: str):
        self.prompt = {
            "role": "ai",
            "content": prompt
        }
        
