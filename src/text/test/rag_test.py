# IO
import os
import sys
import yaml
from pathlib import Path

# Set working directory and system path
os.chdir(Path(__file__).parents[1])
sys.path.append(str(Path(__file__).parents[1]))

# Custom Libraries
from src.text import ModelFactory, RAG
from src.text.prompts.library import retrieval_prompt 

if __name__ == "__main__":
    # Get API Key
    api_key = os.environ.get("OPENAI_API_KEY")

    # Get LLM Model
    factory = ModelFactory(model_name="openai", 
                             api_key=api_key)
    llm = factory.create_model()
    rag_model = RAG()

    # Run model
    data = ["오타니 쇼헤이는 일본 출신의 야구 선수이다.", 
            "오타니 쇼헤이는 미국 메이저리그의 로스앤젤레스 에인절스에서 활약하고 있다.", 
            "오타니 쇼헤이는 투수이자 타자로 활약하고 있다.", 
            "LA 다저스는 올해 역대급 전력을 구축했다.", 
            "LA 다저스는 메이저리그에서 가장 강력한 팀 중 하나로 우승 가능성이 가장 높다."
            ]

    # Search data using query 
    rag_model.set_llm_model(llm_model=llm_model)
    rag_model.fit(documents=data)

    prompt = """
    {name}에 대해 알고 싶어. 
    그에 대해 설명 해줄래? 
    특히, 올해 우승 가능성에 대해 알려줘.
    
    [기본 정보]
    {basic_info}
    
    [상세 정보]
    {related_info}
    """
    query_info = {
        "basic_info": "오타니의 출신지는?",
        "related_info": "LA다저스의 우승 가능성은?"
    }
    answer = rag_model.generate(prompt=prompt, 
                                query_info=query_info,
                                additional_info={"name": "오나티 쇼헤이"}, 
                                top_k=1)

    print("here")