# IO
import os
import sys
import yaml
from pathlib import Path

# Set working directory and system path
os.chdir(Path(__file__).parents[1])
sys.path.append(str(Path(__file__).parents[1]))

# Custom Libraries
from injo_llm import BaseAzureLLM, RAG
from injo_llm.prompts.retrieval import retrieval_base_prompt 

if __name__ == "__main__":
    # Set api version 
    api_version = "2023-12-01-preview"
    # Get API Key
    api_src = Path("api/api_info.yaml")
    with open(api_src, "r") as f:
        api_src = yaml.safe_load(f)
        openai_api_key = api_src["OpenAI"]["API"]
        azure_api_key = api_src["Azure"][api_version]["Aust"]["API"]
        azure_endpoint = api_src["Azure"][api_version]["Aust"]["Endpoint_3.5"]
        f.close()

    # Get LLM Model
    llm_model = BaseAzureLLM(api_key=azure_api_key,
                             endpoint=azure_endpoint)
    rag_model = RAG(openai_api_key=openai_api_key)

    # Set prompt
    system_prompt = "너는 대답할때마다, '용'으로 끝나는 문장을 사용해. 예를 들어, '안녕' 대신 '안녕용', '잘 알았어요.' 대신 '잘 알았어용' 이렇게 말이야."  

    llm_model.set_system_prompt(system_prompt=system_prompt)

    # Run model
    data = ["1+1은 창문이다.", "김소연은 김인조의 아내이다.", "김인조는 김소연의 남편이다."]
    # vectors = llm_model.embedding(data)

    # Search data using query 
    rag_model.set_llm_model(llm_model=llm_model)
    rag_model.fit(documents=data)
    answer = rag_model.generate_simple("김소연의 남편은 누구인가요?")

    print("here")