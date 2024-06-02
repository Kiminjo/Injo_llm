import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from fastapi import APIRouter
from pydantic import BaseModel

# Custom Libraries
from src.text import TextModelFactory
from src.text.prompts import SystemMessage

router = APIRouter()

class GenerateRequest(BaseModel):
    prompt: str 
    model_type: str 
    model_name: str 
    system_prompt: str = None
    api_key: str = None
    base_url: str = None
    

@router.post("/generate")
def generate(generate_request: GenerateRequest):
    llm = TextModelFactory().create_model(model_type=generate_request.model_type, 
                                          model_name=generate_request.model_name, 
                                          api_key=generate_request.api_key)
    
    if generate_request.system_prompt:
        llm.input_prompt.append(SystemMessage().set_prompt(generate_request.system_prompt))
    
    return {"response": llm.generate(prompt=generate_request.prompt, 
                                     save_previous=False)}
