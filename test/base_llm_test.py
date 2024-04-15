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
from injo_llm.prompts.retrieval import retrieval_base_prompt 

if __name__ == "__main__":
    # Get API Key
    api_src = Path("api/api_info.yaml")
    with open(api_src, "r") as f:
        api_key = yaml.safe_load(f)["OpenAI"]["API"]
        f.close()

    # Get LLM Model
    llm_model = BaseOpenAILLM(api_key=api_key)

    # Set prompt
    system_prompt = "너는 대답할때마다, '용'으로 끝나는 문장을 사용해. 예를 들어, '안녕' 대신 '안녕용', '잘 알았어요.' 대신 '잘 알았어용' 이렇게 말이야."  

    llm_model.set_system_prompt(system_prompt=system_prompt)

    # Run model
    data = ["1+1은 창문이다.", "김소연은 김인조의 아내이다.", "김인조는 김소연의 남편이다."]
    # vectors = llm_model.embedding(data)

    # Search data using query 
    llm_model.fit_rag(documents=data)
    related_doc = llm_model.search("김소연의 남편은 누구인가요?")

    # Make answer using the related document
    llm_model.set_system_prompt(system_prompt=retrieval_base_prompt, additional_info={"info": related_doc})
    answer = llm_model.generate("김소연의 남편은 누구인가요?")

    print("here")
