from pathlib import Path
import shutil
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils import GenConfig as cfg


class VectorDBFactory:
    """
    Class to encapsulate the logic for creating and loading the Chroma vector database.
    Creates a new database if it doesn't exist, otherwise loads the existing one.
    ---
    Attributes:
    - chroma_dir: Path to the directory where the Chroma vector store is persisted.
    """

    def __init__(self, 
                 chroma_dir: Path = cfg.chroma_dir, 
                 embeddings_model: str = cfg.embedding_model,
                 parquet_file: Path = cfg.parquet_file
                 ):
        # 1. define the embeddings model ready (we will use it in both cases)
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': cfg.emb_device}
        )
        self.chroma_dir = chroma_dir
        self.parquet_file = parquet_file

    def get_vectorDB(self):
        """ Function to get the vector database, if it doesn't exist, create it."""
        # Check if the Chroma vector store already exists
        if not self.chroma_dir.exists():
            print(f"[get_vectorDB] No database detected in {self.chroma_dir}, creating new one...")
            chunks = self._load_and_chunk_data()
            vectorstore = self._save_to_vector_db(chunks)
        else:   # if it exists, load it
            print(f"[get_vectorDB] Loading existing database from {self.chroma_dir}...")
            vectorstore = Chroma(
                persist_directory=str(self.chroma_dir), 
                embedding_function=self.embeddings_model
            )
    
        return vectorstore  

    # ──────────────────────────────────────────────────────────────────────────────
    # ──  AUXILIAR FUNCTIONS  ──────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────────

    def _load_and_chunk_data(self):
        """
        Loads the parquet file, extracts text, and splits it into chunks.
        Returns a list of Document objects ready for embedding.
        """
        print(f"[_load_and_chunk_data] Loading data from {self.parquet_file}")
        
        # 1. load the parquet file into a DataFrame
        df = pd.read_parquet(self.parquet_file)

        # 2. split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # 3. create a list of Document objects
        documents = []
        for _, row in df.iterrows():
            metadata = {
                "source": row.get("url", "unknown"),
                "title": row.get("title", "Pediatric Advice")
            }
            
            chunks = text_splitter.split_text(row['text'])
            
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=metadata))
        # return a list of Document objects ready for embedding
        return documents

    def _save_to_vector_db(self, chunks):
        """
        Create the Chroma vector database from the list of 
        Document chunks and save it to disk.

        ---
        - chunks: List of Document objects containing the text chunks and metadata.
        """
        # if the chroma_dir already exists, we remove it to create a fresh database
        if self.chroma_dir.exists():
            shutil.rmtree(self.chroma_dir)
            print(f"[_save_to_vector_db] Removed old database from {self.chroma_dir}")
            
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings_model,
            persist_directory=str(self.chroma_dir)   # to string some versions of Chroma prefer it that way
        )
        
        print(f"[VectorDBFactory] 🎁 Database created suscessfully in {self.chroma_dir}")
        return vectorstore


