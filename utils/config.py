# src/config.py
from pathlib import Path

class GenConfig:
    """"General chat configuration"""

    # ── Paths ─────────────────────────────────────────────────────────────────
    root_dir    = Path(__file__).resolve().parent.parent    # proyect root directory
    data_dir    = root_dir / "data"             # directory for data files (e.g., parquet)
    chroma_dir  = root_dir / "chroma_db"        # directory for Chroma vector store (ignored in .gitignore)

    # parquet with the infetion advice data (already processed and cleaned)
    parquet_file    = data_dir / "kidshealth_en_parents_infections.parquet"
    env_file        = root_dir  / ".env"        # file for API key

    # ── Model info ────────────────────────────────────────────────────────────
    model_name      = "llama3.1:8b"             # llama3.1:13b, gpt-3.5-turbo, gpt-4, etc. (must be available in your Ollama server)  
    embedding_model = "all-MiniLM-L6-v2"        # embedding model name (compatible with HuggingFaceEmbeddings)
    ollama_url      = "https://yiyuan.tsc.uc3m.es"     # URL of the Ollama server

    # ── Data parameters ───────────────────────────────────────────────────────
    chunk_size      = 1000          # number of characters per chunk
    chunk_overlap   = 150           # number of characters to overlap between chunks
    retrieval_num   = 5             # number of relevant chunks to retrieve for each query


# ──────────────────────────────────────────────────────────────────────────────
# ── Scrapping parameters ──────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
class ScrapingConfig:
    """ Scrapping configuration parameters, used in the Scrapy spider."""

    # ── General parameters ────────────────────────────────────────────────────
    max_articles    = 500       # For a recomended 300-500 docs, set max to 500
    depth_limit     = 3         # How deep the spider will go following links
    depth_priority  = 1         # Prioritize depth (1) over breadth (0)
    concurrent_reqs = 8         # parallel requests (be careful with too high values, can cause bans)
    min_length      = 200       # Minimum length of text to consider (after extraction)

    name            = 'kidshealth_en_kids'  # name of the spider
    allowed_domains = ['kidshealth.org']   # domains to allow for crawling
    start_urls      = ['https://kidshealth.org/en/parents/infections/']
    parquet_file    = GenConfig.data_dir / "kidshealth_en_parents_infections.parquet"  # where to save the scraped data
    jsonl_path      = GenConfig.data_dir / "kidshealth_en_parents_infections.jsonl"   # also save in jsonl for more raw access
    # ── filtering parameters ──────────────────────────────────────────────────
    # ──────────────────────────── KEYWORDS ────────────────────────────────────
    # this are the need keywords we use as filter for the scrapped articles, 
    # they are related to pediatric infections
    # They are very common words in pediatric texts → not a very restrictive filter.
    keywords = [
        'child', 'children', 'kid', 'kids', 'baby', 'infant',
        'fever', 'disease', 'symptom', 'treatment', 'infection', 'infections', 'ache', 'cough',
        'health', 'doctor', 'hospital', 'pain', 'vaccine', 
        'bacteria', 'bacterial', 'virus', 'viral', 'antibiotic', 'antibiotics', 'flu',
        'bronchitis', 'cold', 'colds', 'temperature', 'poison', 'poisonings', 'gastroenteritis', 
        'inflammation', 'meningitis', 'pneumonia', 'salmonella', 'sepsis', 'otitis',
        'contagious', 'diarrhea', 'conjunctivitis', 
    ]