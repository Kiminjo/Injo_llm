# IO
import os
import sys

from pathlib import Path

import warnings 

warnings.filterwarnings("ignore")

# Set working directory and system path
os.chdir(Path(__file__).parents[1])
sys.path.append(str(Path(__file__).parents[1]))

# Custom Libraries
from injo_llm import ModelFactory, RAG
from injo_llm.prompts import SystemMessage

if __name__ == "__main__":
    # Get API Key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")

    # Get LLM Model
    factory = ModelFactory()
    openai_llm = factory.create_model(model_type="openai", api_key=openai_api_key)
    groq_llm = factory.create_model(model_type="groq", api_key=groq_api_key)
    lmstudio_llm = factory.create_model(model_type="lmstudio", api_key="lmstudio")

    # Set prompt
    system_template = """
    You are the kindergarden teacher for 5 year olds. You are teaching them about the world
    and they have asked you a question. 

    <information>
    President of South Korea: Yoon Suk-yeol
    </information>
    """
    
    system_prompt = SystemMessage(prompt=system_template).prompt

    # Run OpenAI model
    openai_llm.input_messages.append(system_prompt)
    openai_output = openai_llm.generate("Who is the president of South Korea?")

    # Run Groq model
    groq_llm.input_messages.append(system_prompt)
    groq_output = groq_llm.generate("Who is the president of South Korea?")

    # Run LMStudio model
    lmstudio_llm.input_messages.append(system_prompt)
    lmstudio_output = lmstudio_llm.generate("Who is the president of South Korea?")

    print('here')
