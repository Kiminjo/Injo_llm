# IO
import os
import sys

from pathlib import Path

# Set working directory and system path
os.chdir(Path(__file__).parents[1])
sys.path.append(str(Path(__file__).parents[1]))

# Custom Libraries
from src.text import ModelFactory, RAG
from src.text.prompts import UserPrompt, SystemPrompt

if __name__ == "__main__":
    # Get API Key
    api_key = os.environ.get("OPENAI_API_KEY")

    # Get LLM Model
    factory = ModelFactory(model_name="openai", 
                           api_key=api_key)
    llm = factory.create_model()

    # Set prompt
    system_template = "너는 대답할때마다, '용'으로 끝나는 문장을 사용해. 예를 들어, '안녕' 대신 '안녕용', '잘 알았어요.' 대신 '잘 알았어용' 이렇게 말이야."  
    
    sys_creator = SystemPrompt(prompt=system_template)
    system_prompt = sys_creator.set_prompt()

    # Run model
    output = llm.generate("김소연의 남편은 누구인가요?")
