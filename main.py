import os
import sys
from src.chunk_embed_store import get_or_create_vectorstore
from src.chat import start_rag

def main():
    print("=== UC3M PEDIATRIC RAG SYSTEM ===")
    
    # 1. Aseguramos que el vectorstore existe, si no, lo crea una sola vez
    # Esta función debería encapsular la lógica de tu 'chunk_embed_store.py'
    try:
        vectorstore = get_or_create_vectorstore()
    except Exception as e:
        print(f"Error cargando la base de datos: {e}")
        sys.exit(1)
    

    start_rag(vectorstore)


if __name__ == "__main__":
    main()