import os 
from injo_llm.tools import TextModelComparsion
from injo_llm import OpenAILLM, GroqLLM, LMStudioLLM
import pandas as pd 
import yaml 

# XXX for test 
from pathlib import Path 
os.chdir(str(Path(__file__).parent))

def text_model_compare(): 
    text_comparsion = TextModelComparsion()

    # Set api key 
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    lmstudio_api_key = "lm_studio"

    # Set LLM models 
    models = [OpenAILLM(api_key=openai_api_key),
              GroqLLM(api_key=groq_api_key),
              LMStudioLLM(api_key=lmstudio_api_key)]

    # Load test data 
    with open("configs/data.yaml", "r") as f: 
        test_data = yaml.load(f, Loader=yaml.FullLoader)["test_data"]
    
    test_texts = test_data["for_kids"] + test_data["bad_words"]

    # Compare models 
    results = text_comparsion.compare(models=models, test_text=test_texts)
    report = text_comparsion.to_report()
    report.to_excel("report.xlsx")


if __name__=="__main__":
    text_model_compare()