import streamlit as st
from src.chunk_embed_store import get_or_create_vectorstore
from src.chat import get_ai_response  # Importamos la lógica que acabamos de separar

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="BabyChatty AI", page_icon="👶")
st.title("👶 BabyChatty: Pediatric Assistant")
st.caption("Using Llama 3.1 (UC3M) & RAG Logic")
st.markdown("---")

# --- CARGA DE SISTEMA (Cache) ---
@st.cache_resource
def load_db():
    return get_or_create_vectorstore()

vectorstore = load_db()

# --- HISTORIAL DEL CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ENTRADA DEL USUARIO ---
if prompt := st.chat_input("Ask me about pediatric health..."):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Searching medical database..."):
            try:
                # Llamamos a la lógica de chat.py
                answer, num_docs = get_ai_response(prompt, vectorstore)
                
                st.markdown(answer)
                st.caption(f"ℹRetrieved {num_docs} context chunks from database.")
                
                # Guardar respuesta
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Connection Error: {e}")