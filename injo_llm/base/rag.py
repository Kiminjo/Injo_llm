# IO 
import os 
import sys 
from pathlib import Path
import numpy as np
import pickle
from typing import List, Union, Dict

# Vector DB 
import faiss

# Set working directory and system path
os.chdir(Path(__file__).parents[2])
sys.path.append(str(Path(__file__).parents[2]))

# Custom Libraries
from injo_llm.base.base import BaseOpenAILLM
from injo_llm.prompts.retrieval import retrieval_base_prompt
from injo_llm.utils.prompt import fill_prompt

class RAG:
    def __init__(self, llm_model: BaseOpenAILLM = None):
        if llm_model is not None:
            self.llm_model = self.set_llm_model(llm_model)
    
    def fit(self, documents: Union[str, List[str]], db_path: Union[str, Path] = "db/rag.index", document_path: Union[str, Path] = "db/documents.pkl"):
        """
        Train the RAG model
        
        Args:
            - documents: str or List[str]
                The documents for the RAG model
            - db_path: str or Path
                The path to save the database
            - document_path: str or Path
                The path to save the document
        """
        # Convert str type of document to Document type
        if isinstance(documents, str):
            documents = [documents]
        if isinstance(db_path, str):
            db_path = Path(db_path)
        if isinstance(document_path, str):
            document_path = Path(document_path)
        
        # Make save dir 
        db_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.parent.mkdir(parents=True, exist_ok=True)

        # Set embeddings and DB
        embedding_vectors = self.llm_model.embedding(documents)
        faiss_index = faiss.IndexFlatL2(embedding_vectors.shape[1])
        faiss_index.add(embedding_vectors)
        
        # Save the vector store and document as pickle
        faiss.write_index(faiss_index, str(db_path))
        with open(document_path, "wb") as f:
            pickle.dump(documents, f)
            f.close()
    
    def set_llm_model(self, llm_model: BaseOpenAILLM):
        """
        Set the LLM model for the RAG model
        
        Args:
            - llm_model: BaseOpenAILLM
        """
        self.llm_model = llm_model

    def search(self, query: str, db_path: Union[str, Path] = "db/rag.index", document_path: Union[str, Path] = "db/documents.pkl", top_k: int = 5):
        """
        Search the RAG model
        
        Args:
            - query: str
                The query for the search
            - db_path: str or Path
                The path to save the database
            - document_path: str or Path
                The path to save the document
            - top_k: int
                The number of top-k results to return
        
        Returns:
            - results: List[str]
                The top-k results from the search
        """
        # Load the DB and documents
        faiss_index = faiss.read_index(str(db_path))
        with open(document_path, "rb") as f:
            documents = pickle.load(f)
            f.close()

        # Get the embedding from the query
        query_embedding = self.llm_model.embedding(query)
        query_embedding = query_embedding[0]

        # Search the DB
        _, I = faiss_index.search(np.array([query_embedding]), top_k)
        results = [documents[i] for i in I[0] if i != -1]
        return results

    def generate_simple(self, prompt: str, params: Dict = None):
        """
        Generate the answer based on RAGed information 
         
        Args:
            - prompt: str
                The prompt for the answer
            - params: Dict
                The parameters for the answer
        """
        # Search the related documents
        related_doc = self.search(prompt)
        
        # Set the prompt
        self.llm_model.set_system_prompt(system_prompt=retrieval_base_prompt, additional_info={"info": related_doc})
        
        # Generate the answer
        answer = self.llm_model.generate(prompt, params=params)
        return answer
    
    def generate(self, prompt: str, query_info: Dict, additional_info: Dict = None, params: Dict = None):
        """
        Generate the answer from the prompt
        
        Args:
            - prompt: str
                The prompt for the chat
            - query_info: Dict
                The information for the query
            - additional_info: Dict
                The additional information to fill in the prompt
            - params: Dict
                The parameters for the model
                 
                - temperature
                - top_p
        
        Returns:
            - answer: str
                The answer from the model

        Example: 
            prompt = '''
                You have to answer about {topic}.

                [Basic Information]
                {basic_info}

                [Related Information]
                {related_info}
                '''

            additional_info = {"topic": "Othani Shohei"}
            query_info = {
                "basic_info": "Who is Othani Shohei?", 
                "related_info": "How's the Dodgers performing in the 2024 season"
            }

            answer = rag_model.generate(prompt, query_info, additional_info)
        """
        # Set additional information 
        if additional_info is None:
            additional_info = {}

        # Search the related documents
        for key, query in query_info.items():
            related_doc = self.search(query)
            additional_info[key] = "\n".join(related_doc)

        return self.llm_model.generate(prompt, additional_info=additional_info, params=params)
    