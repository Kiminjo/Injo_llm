import os 
from injo_llm.tools import TextModelComparsion
from injo_llm import OpenAILLM, GroqLLM, OllamaLLM, BaseLLM
from injo_llm.prompts import SystemMessage, UserMessage, AIResponseMessage
import yaml 

from pathlib import Path 
os.chdir(str(Path(__file__).parent))

def set_system_prompt(model: BaseLLM,
                      system_prompt_path: str): 

    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()

    sys_instance = SystemMessage()
    model.input_messages.append(sys_instance.set_prompt(system_prompt))
    return model 

def set_fewshots(model: BaseLLM,
                 fewshots: list[str]): 

    user_instance = UserMessage()
    ai_instance = AIResponseMessage()
    for fewshot_idx, fewshot in enumerate(fewshots): 
        if fewshot_idx % 2 == 0:
            model.input_messages.append(user_instance.set_prompt(fewshot))
        else: 
            model.input_messages.append(ai_instance.set_prompt(fewshot))
    return model

def set_system_and_fewshots(model: BaseLLM,
                            system_prompt_path: str,
                            fewshots: list[str]): 

    model = set_system_prompt(model, system_prompt_path)
    model = set_fewshots(model, fewshots)
    return model

def single_turn_text_model_compare(agenda: str = "liar"): 
    text_comparsion = TextModelComparsion()

    # Load system prompts and fewshots 
    with open("configs/configs.yaml", "r") as f: 
        configs = yaml.load(f, Loader=yaml.FullLoader)
    
    # Set system prompts
    system_prompt = configs["agenda"][agenda]["prompt"]
    fewshots = configs["agenda"][agenda]["fewshots"]

    # Set api key 
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    ollama_api_key = "ollama"

    # Set LLM models 
    models = []

    for model_dict in configs["models"]:
        model_type = model_dict["type"]
        model_name = model_dict["model"]        

        if model_type == "openai":
            model = OpenAILLM(chat_model=model_name, api_key=openai_api_key)
        elif model_type == "groq":
            model = GroqLLM(chat_model=model_name, api_key=groq_api_key)
        elif model_type == "ollama":
            model = OllamaLLM(chat_model=model_name, api_key=ollama_api_key)

        model = set_system_and_fewshots(model,
                                        system_prompt_path=system_prompt,
                                        fewshots=fewshots)
        models.append(model)

    # Load test data 
    with open("configs/data.yaml", "r") as f: 
        test_data = yaml.load(f, Loader=yaml.FullLoader)["test_data"]

    test_data_type = configs["data"]["type"]
    
    test_texts = []
    test_texts += [test_data[test_type] for test_type in test_data_type]
    test_texts = test_texts[0]

    # Compare models 
    results = text_comparsion.compare(models=models, test_text=test_texts)
    report = text_comparsion.to_report()
    report.to_excel(configs["output"]["path"])

if __name__=="__main__":
    agenda = "liar"
    single_turn_text_model_compare(agenda=agenda)