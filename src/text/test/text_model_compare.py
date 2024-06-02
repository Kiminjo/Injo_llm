import os 
from injo_llm.tools import TextModelComparsion
from injo_llm import OpenAILLM, GroqLLM, LMStudioLLM

compare_text_model = TextModelComparsion()

openai_api_key = os.environ.get("OPENAI_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")
lmstudio_api_key = "lm-studio"

models = [OpenAILLM(api_key=openai_api_key), 
          GroqLLM(api_key=groq_api_key),
          LMStudioLLM(api_key=lmstudio_api_key)]

test_text = "What is the capital of the United States?"
results = compare_text_model.compare(models=models, test_text=test_text)
report = compare_text_model.to_report()
print(report)