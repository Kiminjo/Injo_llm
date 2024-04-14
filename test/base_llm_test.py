# IO
import os
import sys
import yaml
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
    q1 = "안녕 반가워. 내 이름은 {name}이야."
    a1 = llm_model.generate(q1, additional_info={"name": "소연"})
    print(q1)
    print(a1)

    q2 = "1 + 1은 뭐야?"
    a2 = llm_model.generate(q2)
    print(q2)
    print(a2)

    q3 = "틀렸어. 1+1은 창문이야. 기억해둬"
    a3 = llm_model.generate(q3)
    print(q3)
    print(a3)

    q4 = "1+1은 뭐라고?"
    a4 = llm_model.generate(q4)
    print(q4)
    print(a4)
    print("here")
