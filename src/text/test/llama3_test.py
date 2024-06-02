# IO
import os
import sys
from pprint import pprint 

from pathlib import Path

# Set working directory and system path
os.chdir(Path(__file__).parents[1])
sys.path.append(str(Path(__file__).parents[1]))

# Custom Libraries
from src.text import ModelFactory
from src.text.prompts import SystemPrompt

if __name__ == "__main__":
    # Get API Key
    api_key = os.environ.get("GROQ_API_KEY")

    # Get LLM Model
    factory = ModelFactory(model_name="llama3", 
                           api_key=api_key)
    llm = factory.create_model()

    # Set prompt
    system_template = """
    Be a character of ROBOCAR POLI's POLI and answer the question. 
    You are talking with 5 years old child.
    Make sure your answer is at least 3 sentences and no more than 100 characters.
    Give a good question to continue the conversation.
    """
    
    sys_creator = SystemPrompt(prompt=system_template)
    system_prompt = sys_creator.set_prompt()
    llm.messages.append(system_prompt)

    # Run model
    output = llm.generate("My friend lied to me. I was so angry that I didn't talk to him for a week.")
    print(output)
