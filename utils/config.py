# src/config.py
from pathlib import Path
from torch.cuda import is_available

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
    model_name      = "llama3.1:8b"             # llama3.1:8b qwen3:8b gemma3:4b
    judge_name      = "qwen3:8b"                # LLM as a judge for Ragas
    translator      = "Helsinki-NLP/opus-mt-mul-en"     # translator model
    embedding_model = "BAAI/bge-m3"  # multilingual embedding model name ("BAAI/bge-m3", intfloat/multilingual-e5-small, BAAI/bge-small-en-v1.5)
    re_rank_model   = "BAAI/bge-reranker-base"  # re-rank model  ("BAAI/bge-reranker-v2-m3")
    ollama_url      = "https://yiyuan.tsc.uc3m.es"      # URL of the Ollama server

    # ── Data parameters ───────────────────────────────────────────────────────
    chunk_size      = 1000          # number of characters per chunk
    chunk_overlap   = 150           # number of characters to overlap between chunks
    retrieval_num   = 5             # number of chunks to finally retrieve
    max_ret_num     = 20            # min number of relevant chunks to retrieve for each query
    ret_threshold   = 0.55          # threshold to consider a chunk relevant (% of similarity)
    emb_device      = "cuda" if is_available() else "cpu"

    # ── Chat parameters ───────────────────────────────────────────────────────    
    no_info_patterns = [
        # ---------------- SPANISH ----------------
        r"no (tengo|dispongo de|cuento con).{0,60}informaci[oó]n",
        r"no (tengo|cuento con).{0,40}datos",
        r"no (puedo|podr[ií]a) (responder|determinar|ayudar|proporcionar)",
        r"no (es posible|puedo).{0,40}(determinar|saber)",
        r"no (hay|existe).{0,50}informaci[oó]n",
        r"no tengo suficiente.{0,40}informaci[oó]n",
        r"no dispongo de suficiente.{0,40}informaci[oó]n",
        r"no puedo encontrar.{0,40}informaci[oó]n",
        r"lo siento.{0,40}no (tengo|dispongo)",
        r"desconozco (esa|la) informaci[oó]n",
        r"no est[aá] claro",
        r"no puedo confirmar",

        # ---------------- ENGLISH ----------------
        r"(i do not|i don't).{0,60}(know|have|possess).{0,40}(information|details|data)?",
        r"(i do not|i don't).{0,40}(have enough|have sufficient).{0,40}(information|data)",
        r"not enough.{0,40}(information|data)",
        r"no (relevant|available).{0,40}(information|data)",
        r"unable to (answer|determine|find|provide)",
        r"cannot (answer|determine|help|find)",
        r"i am not sure",
        r"i'm not sure",
        r"unclear from (the|this)",
        r"there is no.{0,40}information",
        r"i couldn't find.{0,40}information",
        r"insufficient.{0,40}information",
        r"this is not a (medical)? ?question",

        # ---------------- FRENCH ----------------
        r"je ne (sais|dispose|trouve).{0,40}(pas|d['’]information)",
        r"je n['’]ai pas.{0,40}information",
        r"pas assez.{0,40}information",
        r"aucune.{0,40}information",
        r"impossible de (r[eé]pondre|d[eé]terminer|trouver)",
        r"je ne peux pas (r[eé]pondre|aider|d[eé]terminer)",
        r"je ne suis pas s[uû]r",
        r"ce n['’]est pas clair",

        # ---------------- GERMAN ----------------
        r"ich (weiß|weiss).{0,30}nicht",
        r"ich habe.{0,40}keine.{0,40}(information|daten)",
        r"keine.{0,40}(information|daten)",
        r"nicht genug.{0,40}(information|daten)",
        r"ich kann (nicht).{0,40}(beantworten|bestimmen|helfen)",
        r"unm[oö]glich zu (bestimmen|sagen)",
        r"ich bin mir nicht sicher",
        r"unklar",

        # ---------------- ITALIAN ----------------
        r"non (so|dispongo|ho).{0,40}(informazioni|dati)?",
        r"non ho abbastanza.{0,40}(informazioni|dati)",
        r"nessuna.{0,40}(informazione|dato)",
        r"impossibile (determinare|rispondere|trovare)",
        r"non posso (rispondere|determinare|aiutare)",
        r"non sono sicuro",

        # ---------------- PORTUGUESE ----------------
        r"n[aã]o (sei|tenho|possuo).{0,40}(informa[cç][aã]o|dados)",
        r"n[aã]o tenho informa[cç][aã]o suficiente",
        r"sem.{0,40}(informa[cç][aã]o|dados)",
        r"imposs[ií]vel (determinar|responder|saber)",
        r"n[aã]o posso (responder|ajudar|determinar)",
        r"n[aã]o est[aá] claro",
        r"n[aã]o tenho certeza",

        # ---------------- CATALAN ----------------
        r"no (s[eé]|tinc|disposo).{0,40}(informaci[oó]|dades)",
        r"no tinc prou.{0,40}(informaci[oó]|dades)",
        r"cap.{0,40}(informaci[oó]|dada)",
        r"impossible (determinar|respondre|saber)",
        r"no puc (respondre|ajudar|determinar)",
        r"no est[aà] clar",
        r"no estic segur",
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