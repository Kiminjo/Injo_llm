import os 
from src.text.tools import TextModelComparsion
from src.text import TextModelFactory

compare_text_model = TextModelComparsion()

openai_api_key = os.environ.get("OPENAI_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")

# Get LLM Model
openai_llm = TextModelFactory().create_model(model_type="openai", 
                                             model_name="gpt-3.5-turbo",
                                             api_key=openai_api_key)

groq_llm = TextModelFactory().create_model(model_type="groq",
                                           model_name="llama3-70b-8192",
                                           api_key=groq_api_key)

models = [openai_llm, groq_llm]

test_text = "What is the capital of the United States?"
results = compare_text_model.compare(models=models, test_text=test_text)
report = compare_text_model.to_report()
print(report)