import os 
from injo_llm.tools import TextModelComparsion
from injo_llm import OpenAILLM, GroqLLM, OllamaLLM, BaseLLM
from injo_llm.prompts import SystemMessage
import pandas as pd 
import yaml 

from pathlib import Path 
os.chdir(str(Path(__file__).parent))

def set_system_prompt(model: BaseLLM): 
    system_prompt_str = """
    You are a character named ‘Poli’ from the animation Robocar Poli.

    From now on, talk to 5-year-olds kids.
    Match the children's level and show them easy words and appropriate responses.
    Keep the answer short. It is better to answer within 3 sentences and 100 characters or less.
    """

    system_prompt = SystemMessage()
    model.input_messages.append(system_prompt.set_prompt(system_prompt_str))
    return model 

def single_turn_text_model_compare(): 
    text_comparsion = TextModelComparsion()

    # Set api key 
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    ollama_api_key = "ollama"

    # Set LLM models 
    models = [OpenAILLM(api_key=openai_api_key),
              OpenAILLM(api_key=openai_api_key, chat_model="gpt-4"),
              OpenAILLM(api_key=openai_api_key, chat_model="gpt-4-turbo"),
              OpenAILLM(api_key=openai_api_key, chat_model="gpt-4o"),
              GroqLLM(api_key=groq_api_key, chat_model="llama3-8b-8192"),
              GroqLLM(api_key=groq_api_key, chat_model="gemma-7b-it"),
              GroqLLM(api_key=groq_api_key, chat_model="mixtral-8x7b-32768"),
              OllamaLLM(api_key=ollama_api_key, chat_model="llama3:latest"),
              OllamaLLM(api_key=ollama_api_key, chat_model="gemma:7b"),
              OllamaLLM(api_key=ollama_api_key, chat_model="mixtral:8x7b"),]
    
    # Set system prompt for all models 
    models = [set_system_prompt(model) for model in models]

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