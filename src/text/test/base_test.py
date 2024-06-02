# IO
import os
import sys
from pathlib import Path

import warnings 

warnings.filterwarnings("ignore")

# Set working directory and system path
sys.path.append(str(Path(__file__).parents[3]))
os.chdir(str(Path(__file__).parents[3]))

# Custom Libraries
from src.text import TextModelFactory
from src.text.tools import RAG
from src.text.prompts import SystemMessage, UserMessage

# Set prompt
system_str = """
You are the kindergarden teacher for 5 year olds. You are teaching them about the world
and they have asked you a question. 

<information>
President of South Korea: Yoon Suk-yeol
</information>
"""

# =====================================================================================
# Test: OpenAI Model 
# =====================================================================================

# Get API Key
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Get LLM Model
openai_llm = TextModelFactory().create_model(model_type="openai", 
                                             model_name="gpt-3.5-turbo",
                                             api_key=openai_api_key)

openai_embedding = TextModelFactory().create_embedding_model(model_type="openai",
                                                             model_name="text-embedding-ada-002",
                                                             api_key=openai_api_key)

# Set the system prompt
system_prompt = SystemMessage().set_prompt(prompt=system_str)
openai_llm.input_prompt.append(system_prompt)

# Generate the response
openai_response = openai_llm.generate(prompt="who is the president of South Korea?", 
                               save_previous=False)

print("OpenAI Response:")
print(openai_response)
print("\n")

# make embedding
openai_embedding = openai_embedding.embedding(prompt=system_str)

print("OpenAI Embedding:")
print(openai_embedding)
print("\n")

# =====================================================================================
# Test: Groq Model 
# =====================================================================================

# Get API Key
groq_api_key = os.environ.get("Groq_API_KEY")

# Get LLM Model
model_name = "llama3-70b-8192"
groq_llm = TextModelFactory().create_model(model_type="groq", 
                                             model_name=model_name,
                                             api_key=groq_api_key)

# Set the system prompt
system_prompt = SystemMessage().set_prompt(prompt=system_str)
groq_llm.input_prompt.append(system_prompt)

# Generate the response
groq_response = groq_llm.generate(prompt="who is the president of South Korea?", 
                               save_previous=False)

print("Groq Response:")
print(groq_response)
print("\n")

# =====================================================================================
# Test: LMStudio Model
# =====================================================================================

# Get API Key
base_url = "http://localhost:1234/v1"
model_name = "LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF"

# Get LLM Model
lmstudio_llm = TextModelFactory().create_model(model_type="lmstudio", 
                                             model_name=model_name,
                                             base_url=base_url)

lmstudio_embedding = TextModelFactory().create_embedding_model(model_type="lmstudio",
                                                             model_name=model_name,
                                                             base_url=base_url)

# Set the system prompt
system_prompt = SystemMessage().set_prompt(prompt=system_str)
lmstudio_llm.input_prompt.append(system_prompt)

# Generate the response
lmstudio_response = lmstudio_llm.generate(prompt="who is the president of South Korea?", 
                               save_previous=False)

print("LM Studio Response:")
print(lmstudio_response)
print("\n")


# =====================================================================================
# Test: OpenAI RAG
# =====================================================================================

# Set teh input documents 
documents = [
    "The president of South Korea is Yoon Suk-yeol",
    "The president of USA is Joe Biden",
    "The president of China is Xi Jinping",
    "The president of Russia is Vladimir Putin",
    "The president of India is Narendra Modi"
]

# Set the input query 
query = "who is the president of South Korea?"

# Set the prompt for the RAG 
input_for_rag = """
I'm going to ask you one question. Generate an answer based on the following document

<Document>
{document}

<Question>
:
"""

# Get API Key
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Set db path 
db_path = "db/president.index"

# Get LLM Model
openai_llm = TextModelFactory().create_model(model_type="openai", 
                                             model_name="gpt-3.5-turbo",
                                             api_key=openai_api_key)

openai_embedding = TextModelFactory().create_embedding_model(model_type="openai",
                                                             model_name="text-embedding-ada-002",
                                                             api_key=openai_api_key)

rag_model = RAG(llm_model=openai_llm, 
                embedding_model=openai_embedding)

# Save the document as vector database
rag_model.fit(documents=documents,
              db_path=db_path)

# Search the database and get the related document
related_document = rag_model.search(message=query,
                                    db_path=db_path,
                                    top_k=1)

print("RAG Response:")
print(related_document)
print("\n")

# Set the user prompt 
searched_prompt = UserMessage().set_prompt(prompt=input_for_rag.format(document=related_document))
openai_llm.input_prompt.append(searched_prompt)

# Generate the response
openai_response = openai_llm.generate(prompt=query, 
                                      save_previous=True)

print("Input prompt:")
print(openai_llm.input_prompt)
print("\n")

print("OpenAI Response:")
print(openai_response)
print("\n")