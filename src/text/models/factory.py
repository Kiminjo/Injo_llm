from src.text import (
    OpenAITextLLM, 
    OllamaTextLLM, 
    GroqTextLLM, 
    LMStudioTextLLM, 
    OpenAITextEmbedding, 
    OllamaTextEmbedding, 
    LMStudioTextEmbedding
)


model_registry = {
    "openai": OpenAITextLLM,
    "ollama": OllamaTextLLM,
    "groq": GroqTextLLM,
    "lmstudio": LMStudioTextLLM
}

embedding_registry = {
    "openai": OpenAITextEmbedding,
    "ollama": OllamaTextEmbedding,
    "lmstudio": LMStudioTextEmbedding
}

class TextModelFactory():
    def create_model(self, 
                     model_type: str,
                     **kwargs):
        """
        Create a model instance
        
        Args:
            - model_type: str
                The type of model to create
            - model_name: str
                The chat model name to use
                ex) "gpt-3.5-turbo-1106"
            - api_key: str
                The API key for the model
            - base_url: str
                The base URL for the model 
                Only used for LM Studio
        """
        return model_registry[model_type](**kwargs)
    
    def create_embedding_model(self,
                               model_type: str,
                               **kwargs):
        """
        Create an embedding model instance

        Args:
            - model_type: str
                The type of model to create
            - model_name: str
                The embedding model to use
                ex) "gpt-3.5-turbo-1106"
            - api_key: str
                The API key for the model
            - base_url: str
                The base URL for the model 
                Only used for LM Studio
        """
        return embedding_registry[model_type](**kwargs)

    