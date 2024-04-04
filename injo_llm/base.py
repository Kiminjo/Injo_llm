# Langchain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# IO
from typing import Dict


class TemplateLLM:
    def __init__(
        self, api_key: str, llm_name: str = "openai", model_name: str = "gpt-3.5-turbo"
    ):
        # Set api key
        self.api_key = api_key

        # Set base llm model
        self.llm = self.set_llm(llm_name=llm_name, model_name=model_name)

    def set_llm(self, llm_name: str, model_name: str):
        if llm_name == "openai":
            llm = ChatOpenAI(openai_api_key=self.api_key, model_name=model_name)
        return llm

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = SystemMessagePromptTemplate.from_template(system_prompt)

    def set_human_prompt(self, human_prompt: str):
        self.human_prompt = HumanMessagePromptTemplate.from_template(human_prompt)

    def generate(self, matched_input: Dict):
        chat_template = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.human_prompt]
        )
        self.chain = chat_template | self.llm

        return self.chain.invoke(matched_input)
