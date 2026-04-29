from utils import RAGChat

def main():
    print("=== UC3M PEDIATRIC RAG SYSTEM ===")

    print(f"[main] Starting Baby Chatty...")
    # 1. We construct the RAG chat
    baby_chatty = RAGChat()
    
    print(f"[main] Starting RAG loop...")
    # 2. We start the RAG loop with the vector db ready
    baby_chatty.start()


if __name__ == "__main__":
    main()