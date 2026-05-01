import os, sys
import ollama
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langdetect import detect, LangDetectException

from utils import GenConfig as cfg
from utils import VectorDBFactory

class RAGChat:
    """
    Class to encapsulate the RAG chat logic, useful for both terminal 
    and Streamlit interfaces.

    --- 
    Attributes:
        - vectorDB: Chroma vector daabase instance if not provided, it will be created on initialization
    """

    def __init__(self):
        # 1. Load the .env file to get the API key
        load_dotenv()

        # 2. start the Ollama client
        self.client = ollama.Client(
            host=cfg.ollama_url,
            headers={"X-API-KEY": os.getenv("OLLAMA_API_KEY")},
        )

        # 3. get the vector database ready, if not provided, create it
        print(f"[RAGChat] Loading vector database from {cfg.chroma_dir}...")
        try:
            vectorDB = VectorDBFactory(cfg.chroma_dir).get_vectorDB()
        except Exception as e:
            print(f"[RagChat] Error loading database: {e}")
            sys.exit(1)
        print(f"[RAGChat] Vector database done...")

        self.vectorDB = vectorDB
    

    def start(self):
        """Start the RAG chat loop in the terminal."""

        print(f"[RAGChat] -- UC3M Pediatric Bot Connected ({cfg.model_name}) ---")
        while True:
            user_input = input("\nParent: ")
            if user_input.lower() in ['exit', 'quit']: 
                break

            try:
                # 1. Detect language
                lang = self._detect_language(user_input)

                print(f"[RAGChat] Language detected: {lang}")
                # 2. Get AI response
                answer, docs = self._get_ai_response(user_input, lang)

                print(f"\nAI: {answer}")

                # ! This is a check of a negative answer
                negative_phrases = [
                    "i don't know", 
                    "i dont know", 
                    "no tengo información", 
                    "no lo sé", 
                    "no encontré información",
                ]            
                answer_start = answer.lower()[:100] 
                has_no_info = any(phrase in answer_start for phrase in negative_phrases)

                if not has_no_info and docs:
                    print(f"DEBUG: Chunks recovered: {len(docs)}")
                    print(f"\n📄 Sources used:")
                    seen = set()
                    for doc in docs:
                        url = doc.metadata.get("source", "unknown")
                        title = doc.metadata.get("title", "unknown")
                        if url not in seen:
                            print(f"  - {title}: {url}")
                            seen.add(url)

            except ollama.ResponseError as e:
                print(f"\n[RAGChat] Ollama response error: {e.error}")
            except Exception as e:
                print(f"\n[RAGChat] connection error: {e}")
    

    def _detect_language(self, text: str) -> str:
        """Detect the language of the user input and return the language name."""
        try:
            lang_code = detect(text)
            lang_map = {
                "es": "Spanish",
                "en": "English",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ca": "Catalan",
            }
            return lang_map.get(lang_code, "English")
        except LangDetectException:
            # fallback to english if detection fails (e.g. very short input)
            print("[RAG] Failed to detect language, falling back to English")
            return "English"


    def _get_ai_response(self, user_input, lang: str = "English"):
        """
        Implement the loop of RAG.
        1. Retrieve relevant chunks from the vectorDB
        2. Construct the prompt with the retrieved context and the user question
        3. Call the Ollama API with the constructed prompt and return the response

        ---
        Attributes:
            - user_input: the question from the user
            - lang: the language of the user input
        """

        # 1. Retrieval
        docs = self.vectorDB.similarity_search(user_input, k=cfg.retrieval_num)  # n best chunks

        # 1.1 Add source info to the context for better traceability:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "Unknown")
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[Source {i}: {title} | {source}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        # 2. Build the prompt for the model, we include instructions and context
        prompt_final = f"""You are a professional Pediatric Assistant. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say you don't know.
        If a different topic question is posed (example: "What is the weather today?", "What can I have for breakfast or for lunch?", "What is your favourite color?", etc.) you should answer that you don't have that information.
        Never invent any response; if there is not in the data, answer that you don't know.

        CRITICAL LANGUAGE RULE: The user is writing in {lang}.
        You MUST respond entirely in {lang}. No exceptions.
        Example: if {lang} is Spanish, write the full answer in Spanish.
        Even the disclaimer at the end must be in {lang}.

        DISCLAIMER RULE:
        Only include a medical disclaimer if the answer is based on the provided context (i.e., you are confident the information comes from the retrieved data)
        If the answer is "I don't know" or the information is not in the context, DO NOT include any disclaimer.
        When required, the disclaimer must be a short sentence stating that the information is general and not a substitute for professional medical care.
        (translated to {lang} if it is not English)

        Context: {context}

        Question: {user_input}
        Helpful Answer in {lang}:"""

        # 3. call the model for a response
        response = self.client.chat(
            model=cfg.model_name,
            messages=[{
                "role": "user",
                "content": prompt_final,
            }]
        )
        
        return response['message']['content'], docs