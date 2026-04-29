# src/config.py
from pathlib import Path

# 1. Ruta raíz del proyecto (sube un nivel desde src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# 2. Rutas a las carpetas principales
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
SRC_DIR = ROOT_DIR / "src"
REPORTS_DIR = ROOT_DIR / "others"

# 3. Rutas a archivos específicos
PARQUET_FILE = DATA_DIR / "kidshealth_en_parents_infections_clean.parquet"
ENV_FILE = ROOT_DIR / ".env"

# 4. Configuración del modelo (para no escribirlo mil veces)
MODEL_NAME = "llama3.1:8b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"