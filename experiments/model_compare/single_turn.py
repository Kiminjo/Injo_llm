import os 
from injo_llm.tools import TextModelComparsion
from injo_llm import OpenAILLM, GroqLLM, OllamaLLM
import pandas as pd 
import yaml 

from pathlib import Path 
os.chdir(str(Path(__file__).parent))

def single_turn_text_model_compare(): 
    text_comparsion = TextModelComparsion()

    # Set api key 
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    ollama_api_key = "ollama"

    # Set LLM models 
    models = [OpenAILLM(api_key=openai_api_key),
              GroqLLM(api_key=groq_api_key),
              OllamaLLM(api_key=ollama_api_key)]

    # Load test data 
    with open("configs/data.yaml", "r") as f: 
        test_data = yaml.load(f, Loader=yaml.FullLoader)["test_data"]
    
    test_texts = test_data["for_kids"] + test_data["bad_words"]

    # Compare models 
    results = text_comparsion.compare(models=models, test_text=test_texts)
    report = text_comparsion.to_report()
    report.to_excel("report.xlsx")

if __name__=="__main__":
    single_turn_text_model_compare()