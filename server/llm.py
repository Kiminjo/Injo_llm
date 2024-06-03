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
    llm_name: str 
    system_prompt: str = None
    api_key: str = None
    base_url: str = None

class EmbeddingRequest(BaseModel):
    prompt: str 
    llm_name: str 
    api_key: str = None
    base_url: str = None
    
@router.post("/{llm_type}/generate")
def generate(llm_type: str, generate_request: GenerateRequest):
    llm = TextModelFactory().create_model(model_type=llm_type, 
                                          model_name=generate_request.llm_name, 
                                          api_key=generate_request.api_key)
    
    if generate_request.system_prompt:
        llm.input_prompt.append(SystemMessage().set_prompt(generate_request.system_prompt))
    
    return {"response": llm.generate(prompt=generate_request.prompt, 
                                     save_previous=False)}

@router.post("/{llm_type}/embedding")
def embedding(llm_type: str, embedding_request: EmbeddingRequest):
    llm = TextModelFactory().create_embedding_model(model_type=llm_type, 
                                                    model_name=embedding_request.llm_name, 
                                                    api_key=embedding_request.api_key)
    
    embedding_vector = llm.embedding(prompt=embedding_request.prompt)
    embedding_vector = embedding_vector.tolist()
    return {"embedding": embedding_vector}