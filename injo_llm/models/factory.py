from injo_llm import OpenAILLM, AzureLLM

model_registry = {
    "openai": OpenAILLM,
    "azure": AzureLLM
}

class ModelFactory:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def create_model(self, **kwargs):
        return model_registry[self.model_name](**{**self.kwargs, **kwargs})
    