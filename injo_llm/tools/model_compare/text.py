from injo_llm import BaseLLM 
import pandas as pd 

class TextModelComparsion():
    def compare(self,
                models: list[BaseLLM],
                test_text: str | list[str]
                ):
        """
        Compare the text model
        
        Args:
            - models: list[BaseLLM]
                The list of models to compare
            - test_text: str | list[str]
                The text to compare
        
        Returns:
            - results: dict
                The results from the comparison
                
        Example:
        >>> compare_text_model = CompareTextModel()
        >>> models = [OpenAILLM(api_key=openai_api_key), 
                        GroqLLM(api_key=groq_api_key),
                        LMStudioLLM(api_key=lmstudio_api_key)
                    ]
        >>> test_text = "What is the capital of the United States?"
        >>> results = compare_text_model.compare(models=models, test_text=test_text)
        """
        # Change the test_text to a list if it is a string
        if isinstance(test_text, str):
            test_text = [test_text]

        self.results = {}
        for model in models:
            for text in test_text:
                answer = model.generate(prompt=text)
                self.results[model.__class__.__name__] = {
                    "prompt": text,
                    "response": answer,
                    "chat_model_name": model.chat_model_name,
                    "latency": model.latency,
                    "input_tokens": model.input_tokens,
                    "output_tokens": model.output_tokens,
                    "total_tokens": model.total_tokens,
                }
        return self.results
    
    def to_report(self): 
        """
        Convert the results to a report as pandas DataFrame
        
        Watched information:
            - response(str): The response from the model
            - chat_model_name(str): The name of the chat model
            - prompt(str): The prompt for the chat
            - latency(float): The time it takes to generate the response
            - input_tokens(int): The number of tokens in the input
            - output_tokens(int): The number of tokens in the output
            - total_tokens(int): The total number of tokens used

        Returns:
            - report: pd.DataFrame
                The report of the comparison

        """
        report = pd.DataFrame(self.results).T
        report.columns = ["prompt", "response", "chat_model_name", "latency", "input_tokens", "output_tokens", "total_tokens"]
        return report
        
        
