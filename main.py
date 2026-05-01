import random
from utils import RAGChat, GenConfig as cfg

def load_random_questions(file_path, n=10):
    """ Utility function to load random questions from a text file for evaluation."""
    with open(file_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    
    return random.sample(questions, min(n, len(questions)))

def main():
    print("=== UC3M PEDIATRIC RAG SYSTEM ===")

    print(f"[main] Starting Baby Chatty...")
    # 1. We construct the RAG chat
    baby_chatty = RAGChat()
    
    print(f"[main] Starting RAG loop...")
    # 2. We start the RAG loop with the vector db ready
    baby_chatty.start(eval_mode=True)

    # evaluation of model via n questions
    # random.seed(42)  # for reproducibility
    # questions = load_random_questions(cfg.eval_questions, n=5)
    # baby_chatty.eval_questions(questions)

if __name__ == "__main__":
    main()