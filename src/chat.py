import os
import ollama
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, MODEL_NAME, EMBEDDING_MODEL

# --- 1. CONFIGURACIÓN DE ENTORNO ---
env_path = Path(".env")
load_dotenv()

ollama_api_key = os.getenv("OLLAMA_API_KEY")
OLLAMA_URL = "https://yiyuan.tsc.uc3m.es"
llm_model = MODEL_NAME

# --- 2. CONFIGURACIÓN DEL CLIENTE OLLAMA ---
client = ollama.Client(
    host=OLLAMA_URL,
    headers={"X-API-KEY": ollama_api_key},
)

def get_ai_response(user_input, vectorstore):
    """
    Esta función contiene la lógica central del RAG. 
    Se usa tanto en la terminal como en la interfaz de Streamlit.
    """
    # 1. RECUPERACIÓN (Retrieval)
    # Buscamos los 5 fragmentos más relevantes
    docs = vectorstore.similarity_search(user_input, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. CONSTRUCCIÓN DEL PROMPT (Tu lógica exacta)
    prompt_final = f"""You are a professional Pediatric Assistant. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say you don't know.
    If a different topic question is posed (example: "What is the weather today?") you should answer that you don't have that information.
    Never invent any response; if there is not in the data, answer that you don't know.

    CRITICAL: Answer strictly in the same language as the input.
    Always add: 'This is not medical advice, please consult a pediatrician.'

    Context: {context}

    Question: {user_input}
    Helpful Answer:"""

    # 3. LLAMADA AL MODELO
    response = client.chat(
        model=llm_model,
        messages=[{
            "role": "user",
            "content": prompt_final,
        }]
    )
    
    return response['message']['content'], docs

def start_rag(vectorstore):
    """
    Bucle de consola tradicional (se mantiene igual para pruebas rápidas).
    """
    print(f"--- UC3M Pediatric Bot Connected ({llm_model}) ---")

    while True:
        user_input = input("\nParent: ")
        if user_input.lower() in ['exit', 'quit']: 
            break

        try:
            # Reutilizamos la función de arriba
            answer, docs = get_ai_response(user_input, vectorstore)
            
            print(f"\nAI: {answer}")
            print(f"DEBUG: Chunks recuperados: {len(docs)}")
            
        except Exception as e:
            print(f"\nError de conexión: {e}")

if __name__ == "__main__":
    # Esto es solo por si ejecutas el archivo directamente para pruebas
    from src.chunk_embed_store import get_or_create_vectorstore
    vs = get_or_create_vectorstore()
    start_rag(vs)