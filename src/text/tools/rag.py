# IO 
import os 
import sys 
from pathlib import Path
import numpy as np
import pickle
from typing import Optional

# Vector DB 
import faiss

# Set working directory and system path
os.chdir(Path(__file__).parents[3])
sys.path.append(str(Path(__file__).parents[3]))

# Custom Libraries
from src.text import BaseTextModel, BaseTextEmbeddingModel


class RAG:
    def __init__(self, 
                 llm_model: BaseTextModel,
                 embedding_model: BaseTextEmbeddingModel
                 ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model

    def fit(self, 
            documents: str | list[str], 
            db_path: str | Path = "db/rag.index", 
            document_path: Optional[str] | Optional[Path] = None
            ):
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

        # Set the default document path
        if document_path is None:
            document_path = db_path.with_suffix(".pkl")
        
        # Make save dir 
        db_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.parent.mkdir(parents=True, exist_ok=True)

        # Set embeddings and DB
        embedding_vectors = self.embedding_model.embedding(documents)
        faiss_index = faiss.IndexFlatL2(embedding_vectors.shape[1])
        faiss_index.add(embedding_vectors)
        
        # Save the vector store and document as pickle
        faiss.write_index(faiss_index, str(db_path))
        with open(document_path, "wb") as f:
            pickle.dump(documents, f)
            f.close()

    def search(self, 
               message: str, 
               db_path: str | Path = "db/rag.index", 
               document_path: Optional[str] | Optional[Path] = None,
               top_k: int = 5
               ):
        """
        Search the RAG model
        
        Args:
            - message: str
                The message for the search
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
        if isinstance(db_path, str):
            db_path = Path(db_path)
        if isinstance(document_path, str):
            document_path = Path(document_path)
            
        # Set the default document path
        if document_path is None:
            document_path = db_path.with_suffix(".pkl")

        # Load the DB and documents
        faiss_index = faiss.read_index(str(db_path))
        with open(document_path, "rb") as f:
            documents = pickle.load(f)
            f.close()

        # Get the embedding from the message
        message_embedding = self.embedding_model.embedding(message)
        message_embedding = message_embedding[0]

        # Search the DB
        _, I = faiss_index.search(np.array([message_embedding]), top_k)
        results = [documents[i] for i in I[0] if i != -1]
        return results
