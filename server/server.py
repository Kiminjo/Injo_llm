from fastapi import FastAPI
from llm import router as llm_router 

app = FastAPI()

app.include_router(llm_router, prefix="/api/v1")