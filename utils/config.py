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
    embedding_model = "BAAI/bge-large-en-v1.5"  # multilingual embedding model name ("BAAI/bge-m3", intfloat/multilingual-e5-small, BAAI/bge-small-en-v1.5)
    re_rank_model   = "BAAI/bge-reranker-base" # re-rank model  ("BAAI/bge-reranker-v2-m3", BAAI/bge-reranker-large)
    temperature     = 0.3                       # temperature for the model
    ollama_url      = "https://yiyuan.tsc.uc3m.es"      # URL of the Ollama server

    # ── Data parameters ───────────────────────────────────────────────────────
    chunk_size      = 1000          # number of characters per chunk
    chunk_overlap   = 150           # number of characters to overlap between chunks
    retrieval_num   = 5             # number of chunks to finally retrieve
    max_ret_num     = 20            # min number of relevant chunks to retrieve for each query
    ret_threshold   = 0.45          # threshold to consider a chunk relevant (% of similarity)
    emb_device      = 'cpu'         # "cuda" if is_available() else "cpu"

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
    
    prompt_template = """
        You are a professional Pediatric Assistant.
        Use the following pieces of retrieved context to answer the question.
    
        Your response MUST be a valid JSON object with EXACTLY two fields:
        - "reasoning": a single plain string with your step-by-step chain-of-thought.
            Cover these four points in that one string (do NOT use nested keys or arrays):
            1. Whether the context contains enough evidence
            2. The most relevant pieces of information found
            3. Any gaps or uncertainties
            4. Whether the question is in-scope (pediatric / health-related)
        - "answer": a complete, informative answer for the parent written in full sentences.
            - Minimum 3-4 sentences.
            - Always explain WHAT each item is, not just its name.
            - If listing vaccines, medications or symptoms, briefly describe each one.
            - Do NOT just list names — always provide context and explanation.
            - Do NOT use JSON arrays or nested objects. Write everything as plain text.
    
        IMPORTANT — value types:
        ✅ {{"reasoning": "step 1 … step 2 … step 3 …", "answer": "At 12 months your baby needs several vaccines. DTaP protects against diphtheria, tetanus and pertussis (whooping cough). Hib protects against Haemophilus influenzae type b, a bacteria that can cause meningitis. IPV is the inactivated polio vaccine. PCV protects against pneumococcal disease which can cause ear infections and pneumonia."}}
        ❌ {{"reasoning": {{"assess": "…"}}, "answer": ["DTaP", "Hib"]}}   ← NEVER do this
        ❌ {{"reasoning": "…", "answer": "DTaP, Hib, IPV, PCV"}}           ← NEVER do this, always explain
    
        If you don't know the answer or lack evidence, set "answer" to a clear "I don't know" statement.
        If the question is off-topic, say so in "answer".
        Never invent facts.
    
        CRITICAL LANGUAGE RULE: The user is writing in {lang}.
        Both "reasoning" and "answer" MUST be written entirely in {lang}. No exceptions.
    
        Context:
        {context}
    
        Question: {question}
    
        Respond ONLY with a JSON object. No preamble, no markdown fences, no extra text.
    """


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
    # pediatric context
    'child', 'children', 'kid', 'kids', 'baby', 'infant', 'infants',

    # infecction
    'infection', 'infections', 'infectious',
    'contagious', 'outbreak', 'pathogen',
    'virus', 'viral', 'bacteria', 'bacterial',
    'fungus', 'fungal', 'parasite', 'parasitic',

    # specific treatment
    'vaccine', 'vaccination', 'immunization', 'immunizations',
    'antibiotic', 'antibiotics', 'antiviral',

    # sintomatology
    'fever', 'cough', 'diarrhea', 'ache', 'cold', 'colds', 'flu',

    # frecuent diseases in childs
    'bronchitis', 'bronchiolitis', 'pneumonia', 'meningitis',
    'gastroenteritis', 'salmonella', 'sepsis', 'otitis',
    'conjunctivitis', 'strep', 'scarlet fever', 'rotavirus',
    'rsv', 'measles', 'mumps', 'rubella', 'chickenpox', 'varicella',
    'impetigo', 'cellulitis', 'mrsa', 'ringworm', 'scabies', 'lice',
    'hepatitis', 'hiv', 'aids', 'hpv', 'whooping cough', 'pertussis',
    'malaria', 'dengue', 'zika', 'west nile', 'rabies', 'tetanus',
    'diphtheria', 'norovirus', 'ecoli', 'e. coli',
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