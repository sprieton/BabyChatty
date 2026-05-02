import random, csv, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from utils import RAGChat, GenConfig as cfg


 
def main():
    print("=== UC3M PEDIATRIC RAG SYSTEM ===")
    print("[main] Starting Baby Chatty...")
 
    baby_chatty = RAGChat()
 
    baby_chatty.start()
 
if __name__ == "__main__":
    main()