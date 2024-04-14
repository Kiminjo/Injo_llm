# IO
import os
import sys
from pathlib import Path

# Set working directory and system path
os.chdir(Path(__file__).parents[1])
sys.path.append(str(Path(__file__).parents[1]))

# Custom Libraries
from injo_llm import BaseOpenAILLM


if __name__ == "__main__":
    # Get API Key
    api_src = Path("api/api.txt")
    with open(api_src, "r") as f:
        api_key = f.read().strip()
        f.close()

    # Get LLM Model
    llm_model = BaseOpenAILLM(api_key=api_key)

    # Set prompt
    system_prompt = "너는 대답할때마다, '용'으로 끝나는 문장을 사용해. 예를 들어, '안녕' 대신 '안녕용' 이렇게 말이야."  

    llm_model.set_system_prompt(system_prompt=system_prompt)

    # Run model
    data = ["1+1은 창문이다.", "김소연은 김인조의 아내이다.", "김인조는 김소연의 남편이다."]
    # vectors = llm_model.embedding(data)

    llm_model.train_rag(documents=data)
    doc = llm_model.search("김소연의 남편은 누구인가요?")
    print("here")
