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

# --- 2. CONFIGURACIÓN DEL CLIENTE OLLAMA (Tal cual tu práctica) ---
client = ollama.Client(
    host=OLLAMA_URL,
    headers={"X-API-KEY": ollama_api_key},
)

def start_rag(vectorstore):
    # 3. CARGAR BASE DE DATOS LOCAL
    # print("Loading local database...")
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    print(f"--- UC3M Pediatric Bot Connected ({llm_model}) ---")

    while True:
        user_input = input("\nParent: ")
        if user_input.lower() in ['exit', 'quit']: break

        # 4. RECUPERACIÓN (Retrieval)
        # Buscamos los 3 fragmentos más relevantes en nuestra DB local
        docs = vectorstore.similarity_search(user_input, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 5. CONSTRUCCIÓN DEL PROMPT (Manual)
        prompt_final = f"""You are a professional Pediatric Assistant. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say you don't know.

        CRITICAL: Answer strictly in the same language as the input.
        Always add: 'This is not medical advice, please consult a pediatrician.'

        Context: {context}

        Question: {user_input}
        Helpful Answer:"""

        # 6. LLAMADA AL MODELO (Usando tu lógica de la uni)
        try:
            response = client.chat(
                model=llm_model,
                messages=[{
                    "role": "user",
                    "content": prompt_final,
                }]
            )
            
            print(f"\nAI: {response['message']['content']}")
            print(f"DEBUG: Chunks recuperados: {len(docs)}") # Debería ser 3
            
        except Exception as e:
            print(f"\nError de conexión: {e}")

if __name__ == "__main__":
    start_rag()