# IO
import os
import sys
import yaml
from pathlib import Path

# Set working directory and system path
os.chdir(Path(__file__).parents[1])
sys.path.append(str(Path(__file__).parents[1]))

# Custom Libraries
from injo_llm import BaseLLM


if __name__ == "__main__":
    # Set DB Path 
    db_path  = Path("db")
    db_path.mkdir(exist_ok=True, parents=True)

    # Get API Key
    with open("api/api_info.yaml", "r") as f:
        api_key = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    api_key = api_key["OpenAI"]["API"]

    # Get Information
    with open("api/info.txt", "r") as f: 
        info = f.read()
        f.close()

    # Get LLM Model
    llm_model = BaseLLM(api_key=api_key)

    # Set RAG Model 
    llm_model.train_rag(documents=info, db_path=db_path)

    # Run model
    llm_model2 = BaseLLM(api_key=api_key)
    
    # Set prompt
    system_prompt = "너는 지금부터 {talk}로 대답해줘."
    human_prompt = "{person}에 대해 설명해줘."

    llm_model2.set_system_prompt(system_prompt=system_prompt)
    llm_model2.set_human_prompt(human_prompt=human_prompt)

    # Set RAG 
    llm_model2.setup_rag(db_path=db_path)
    answer = llm_model2.generate(matched_input={"talk": "존댓말", "person": "오동근"})
    print(answer)
