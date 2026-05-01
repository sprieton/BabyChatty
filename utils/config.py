# src/config.py
from pathlib import Path

class GenConfig:
    """"General chat configuration"""

    # ── Paths ─────────────────────────────────────────────────────────────────
    root_dir    = Path(__file__).resolve().parent.parent    # proyect root directory
    data_dir    = root_dir / "data"             # directory for data files (e.g., parquet)
    chroma_dir  = root_dir / "chroma_db"        # directory for Chroma vector store (ignored in .gitignore)
    reports_dir = root_dir / "reports"          # directory for analysis reports (e.g., corpus audit)

    # parquet with the infetion advice data (already processed and cleaned)
    parquet_file    = data_dir / "kidshealth_en_parents_infections.parquet"
    eval_questions  = data_dir / "eval_questions.csv"   # text file with the question to evaluate the RAG system (one question, can be used for quick tests)
    env_file        = root_dir  / ".env"        # file for API key

    # ── Model info ────────────────────────────────────────────────────────────
    model_name      = "llama3.1:8b"             # qwen3:8b gemma3:4b
    judge_name      = "qwen3:8b"                # LLM as a judge for Ragas
    embedding_model = "all-MiniLM-L6-v2"        # embedding model name (compatible with HuggingFaceEmbeddings)
    ollama_url      = "https://yiyuan.tsc.uc3m.es"     # URL of the Ollama server

    # ── Data parameters ───────────────────────────────────────────────────────
    chunk_size      = 1000          # number of characters per chunk
    chunk_overlap   = 150           # number of characters to overlap between chunks
    retrieval_num   = 5             # number of relevant chunks to retrieve for each query

    # ── Chat parameters ───────────────────────────────────────────────────────    
    no_info_patterns = [
        # Spanish
        r"no (tengo|dispongo de|cuento con).{0,50}información",
        r"no (puedo|podría) (responder|determinar|ayudar)",
        r"no (hay|existe).{0,40}información",

        # English
        r"(i do not|i don't).{0,40}(know|have|possess).{0,40}(information|details)?",
        r"(not enough|no).{0,40}information",
        r"unable to (answer|determine)",
        r"cannot (answer|determine|help)",
        r"this (is) not a (medical)? question",
        
        # French
        r"je ne (sais|dispose).{0,40}(pas|d'information)",
        
        # German
        r"ich (weiß|weiss).{0,20}nicht",
        r"keine.{0,20}information",
        
        # Italian
        r"non (so|dispongo).{0,40}(informazioni)?",
        
        # Portuguese
        r"não (sei|tenho).{0,40}(informação|informações)",
        
        # Catalan
        r"no (sé|tinc).{0,40}(informació)",
    ]

    disclamer_prompt = {
        "Spanish": "Esto no es un consejo médico, por favor consulte a un pediatra.",
        "English": "This is not medical advice, please consult a pediatrician.",
        "French": "Ceci n'est pas un avis médical, veuillez consulter un pédiatre.",
        "German": "Dies ist keine medizinische Beratung, bitte konsultieren Sie einen Kinderarzt.",
        "Italian": "Questo non è un consiglio medico, si prega di consultare un pediatra.",
        "Portuguese": "Isto não é aconselhamento médico, por favor consulte um pediatra.",
        "Catalan": "Això no és un consell mèdic, si us plau consulteu un pediatre."
        }



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

class AnalysisConfig:
    """ Configuration for the corpus analysis and auditing."""
    # ── Analysis parameters ───────────────────────────────────────────────────
    min_word_count = 80         # Minimum word count to consider a document valid (after extraction)
    boilerplate_threshold = 0.5 # Threshold for considering a document as possible boilerplate
    corpus_name = "parents_infections"  # name of the corpus to analyze (key in corpora dict)

    # ── Analysis paths ────────────────────────────────────────────────────────
    parquet_file = GenConfig.data_dir / "kidshealth_en_parents_infections_clean.parquet"  # parquet file to analyze
    out_path = GenConfig.reports_dir / "corpus_comparison_summary.csv"  # where to save the cleaned parquet
    corpus_quality_path = GenConfig.reports_dir / "final_corpus_quality_summary.csv"  # where to save the corpus quality metrics (one row per corpus)
    corpus_rew_30 = GenConfig.reports_dir / "corpus_manual_review_sample.csv"  # where to save the manual review sample (30 random docs)
    # ── Analysis extra constructs ─────────────────────────────────────────────
    # dictionary of corpora to compare, can add more parquets here with a name as key
    corpora = { "parents_infections": GenConfig.parquet_file } 

    boilerplate_patterns = [
        r"Reviewed by:",
        r"Medically reviewed by:",
        r"Nemours KidsHealth",
        r"©",
        r"for Parents",
        r"for Kids",
        r"for Teens",
        r"Listen",
        r"Print",
        r"en español",
        r"More on this topic",
    ]

    non_clinical_url_patterns = [
        r"/about\.html$",
        r"/all-categories\.html$",
    ]

    non_clinical_title_patterns = [
        r"^About Nemours KidsHealth",
        r"^Health Topics for Parents$",
    ]