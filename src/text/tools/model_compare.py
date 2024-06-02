from src.text import BaseTextModel
import pandas as pd 
from tqdm import tqdm
from collections import defaultdict

class TextModelComparsion():
    def compare(self,
                models: list[BaseTextModel],
                test_text: str | list[str]
                ):
        """
        Compare the text model
        
        Args:
            - models: list[BaseTextModel]
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

        self.results = defaultdict(list)
        for model_idx, model in enumerate(models):
            print(f"{model_idx + 1}/{len(models)}  Model: {model.__class__.__name__}  Model Name: {model.chat_model_name} Test Progress...")
            
            for text in tqdm(test_text, total=len(test_text)):
                answer = model.generate(prompt=text)
                self.results[model.__class__.__name__].append({
                    "chat_model_name": model.model_name,
                    "prompt": text,
                    "response": answer,
                    "latency": model.latency,
                    "input_tokens": model.input_tokens,
                    "output_tokens": model.output_tokens,
                    "total_tokens": model.total_tokens,
                })
                # Remove the user prompt from the input messages
                model.clear()
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
        if hasattr(self, "results"):
            report = pd.DataFrame()
            
            for model_name, results in self.results.items():
                df = pd.DataFrame(results)
                df["model_name"] = model_name
                report = pd.concat([report, df])
            
            return report
        
        
