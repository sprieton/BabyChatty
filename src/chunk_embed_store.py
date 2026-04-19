import os
import shutil
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Importamos las rutas y modelos desde tu nuevo config
from src.config import PARQUET_FILE, CHROMA_DIR, EMBEDDING_MODEL

def load_and_chunk_data(file_path):
    """Carga el parquet y lo divide en fragmentos."""
    print(f"Cargando datos desde: {file_path}")
    
    # Usamos el archivo definido en config si no se pasa otro
    df = pd.read_parquet(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    documents = []
    for _, row in df.iterrows():
        metadata = {
            "source": row.get("url", "unknown"),
            "title": row.get("title", "Pediatric Advice")
        }
        
        chunks = text_splitter.split_text(row['text'])
        
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata=metadata))

    return documents

def save_to_vector_db(chunks):
    """Crea la base de datos vectorial desde cero."""
    # CHROMA_DIR viene de config.py (es un objeto Path)
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print(f"Base de datos antigua en {CHROMA_DIR} eliminada.")

    print(f"Generando embeddings con {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Convertimos CHROMA_DIR a string porque algunas versiones de Chroma lo prefieren así
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    
    print(f"¡Éxito! Base de datos vectorial guardada en: {CHROMA_DIR}")
    return vectorstore

def get_or_create_vectorstore():
    """Lógica principal: si existe la carga, si no la crea."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if not CHROMA_DIR.exists():
        print("No se detectó base de datos. Iniciando proceso de ingesta...")
        chunks = load_and_chunk_data(PARQUET_FILE)
        vectorstore = save_to_vector_db(chunks)
    else:
        print(f"Cargando base de datos existente desde {CHROMA_DIR}...")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR), 
            embedding_function=embeddings
        )
    
    return vectorstore

# Esto permite que si ejecutas este script solo, se genere la DB
if __name__ == "__main__":
    get_or_create_vectorstore()