import os
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])
print("Load LLM Model...")

tools = load_tools(["wikipedia", "llm-math"], 
                   llm=llm)
print("Load Tools...")

agent  = initialize_agent(tools=tools, 
                          llm=llm, 
                          agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                          verbose=True,
                          handle_parsing_errors=True)
print("Initialize Agent...")

result = agent.run("What is the age of Korean MC Yoo Jae-suk? Multiply that age by 10 and add 5.")

print("\n \n")
print("-" * 50)
print(result)