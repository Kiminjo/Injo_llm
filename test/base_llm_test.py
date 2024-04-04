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
    # Get API Key
    with open("api/api_info.yaml", "r") as f:
        api_key = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    api_key = api_key["OpenAI"]["API"]

    # Get LLM Model
    llm_model = BaseLLM(api_key=api_key)

    # Set prompt
    system_prompt = "너는 지금부터 {talk}로 대답해줘."
    human_prompt = "{city}에 대해 설명해줘."

    llm_model.set_system_prompt(system_prompt=system_prompt)
    llm_model.set_human_prompt(human_prompt=human_prompt)

    # Run model
    answer = llm_model.generate(matched_input={"talk": "반말", "city": "서울"})
    print(answer)
