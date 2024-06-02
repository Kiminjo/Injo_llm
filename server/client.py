import requests
import os 

url = "http://localhost:8000/api/v1/generate"
api_key = os.environ.get("OPENAI_API_KEY")

chat_input = {
    "api_key": api_key,
    "model_type": "openai",
    "model_name": "gpt-3.5-turbo-1106",
    "prompt": "who is the president of the united states?"
}

response = requests.post(url, json=chat_input)
print(response.json())