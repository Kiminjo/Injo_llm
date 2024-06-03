import requests
import os 

llm_type = "openai"

url = f"http://localhost:8000/api/v1/{llm_type}/generate"

if llm_type == "openai":
    api_key = os.environ.get("OPENAI_API_KEY")
    chat_input = {
        "api_key": api_key,
        "llm_name": "gpt-3.5-turbo-1106",
        "prompt": "who is the president of the united states?"
    }

elif llm_type == "groq":
    api_key = os.environ.get("GROQ_API_KEY")
    chat_input = {
        "api_key": api_key,
        "llm_name": "llama3-70b-8192",
        "prompt": "who is the president of the united states?"
    }

ai_response = requests.post(url, json=chat_input)

print("AI Response: ")
print(ai_response.json())
print("\n")

if llm_type == "openai":
    url = f"http://localhost:8000/api/v1/{llm_type}/embedding"
    embedding_input = {
        "api_key": api_key,
        "llm_name": "text-embedding-3-small",
        "prompt": "who is the president of the united states?"
    }
    
    embedding_response = requests.post(url, json=embedding_input)
    
    print("Embedding Response: ")
    print(embedding_response.json())
    print("\n")
    
    