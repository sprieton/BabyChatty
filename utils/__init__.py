# __init__.py for make utils a package and to centralize configuration management

from .config import GenConfig, ScrapingConfig, AnalysisConfig
from .vectorDB_factory import VectorDBFactory
from .chat import RAGChat