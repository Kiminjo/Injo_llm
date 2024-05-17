from injo_llm import OpenAILLM, AzureLLM, GroqLLM, LMStudioLLM, OllamaLLM

model_registry = {
    "openai": OpenAILLM,
    "azure": AzureLLM,
    "groq": GroqLLM,
    "lmstudio": LMStudioLLM,
    "ollama": OllamaLLM
}

class ModelFactory():
    def create_model(self, 
                     model_type: str,
                     **kwargs):
        """
        Create a model instance
        
        Args:
            - model_type: str
                The type of model to create
            - chat_model: str
                The chat model name to use
                ex) "gpt-3.5-turbo-1106"
            - embedding_model: str
                The embedding model name to use
                ex) "text-embedding-3-small"
            - api_key: str
                The API key for the model
            - base_url: str
                The base URL for the model 
                Only used for LM Studio
        """
        return model_registry[model_type](**kwargs)
    