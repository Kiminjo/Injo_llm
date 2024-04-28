# Langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# IO
from typing import Dict, Union
from pathlib import Path


class LangchainBaseLLM:
    def __init__(self, 
                 api_key: str, 
                 llm_name: str = "openai", 
                 model_name: str = "gpt-3.5-turbo"
                 ):
        # Set params for class
        self.api_key = api_key
        self.chatbot = None

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

    def train_rag(self, documents: str, db_path: Union[str, Path]):
        # Convert str type of document to Document type
        documents = Document(documents)

        # Set embeddings and DB
        embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        db = FAISS.from_documents([documents], embeddings)

        # Save the vector store
        db.save_local(db_path)

    def setup_rag(self, db_path: Union[str, Path]):
        # Load the vector store
        embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

        # Set the retriever
        retriever = db.as_retriever()
        self.chatbot = RetrievalQA.from_llm(
            llm=self.llm, retriever=retriever, return_source_documents=True
        )

    def generate(self, matched_input: Dict):
        if self.chatbot is None:
            chat_template = ChatPromptTemplate.from_messages(
                [self.system_prompt, self.human_prompt]
            )
            self.message = chat_template.format_messages(**matched_input)

            answer = self.llm.invoke(self.message)
        else:
            system_message = self.system_prompt.format_messages(**matched_input)[0].content
            human_message = self.human_prompt.format_messages(**matched_input)[0].content
            self.message = f"{system_message}\n{human_message}"
            answer = self.chatbot(self.message)

        return answer
