import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import trafilatura
import pandas as pd

from src.config import DATA_DIR

# ---------------------------------------------------------------------------
# Configuración principal — ajusta estos valores si necesitas
# ---------------------------------------------------------------------------
MAX_ARTICLES = 500       # Para 300-500 docs, ponemos el techo en 500
DEPTH_LIMIT = 4          # Profundidad de navegación desde la URL raíz
CONCURRENT_REQUESTS = 8  # Peticiones en paralelo (no subas mucho para no saturar)

# ---------------------------------------------------------------------------
# Keywords pediátricas en inglés
# Si un artículo no contiene ninguna de estas palabras, se descarta.
# Son palabras muy frecuentes en textos de pediatría → filtro poco restrictivo.
# ---------------------------------------------------------------------------
KEYWORDS = [
    'child', 'children', 'kid', 'kids', 'baby', 'infant',
    'fever', 'disease', 'symptom', 'treatment', 'infection', 'infections', 'ache', 'cough',
    'health', 'doctor', 'hospital', 'pain', 'vaccine', 
    'bacteria', 'bacterial', 'virus', 'viral', 'antibiotic', 'antibiotics', 'flu',
    'bronchitis', 'cold', 'colds', 'temperature', 'poison', 'poisonings', 'gastroenteritis', 
    'inflammation', 'meningitis', 'pneumonia', 'salmonella', 'sepsis', 'otitis',
    'contagious', 'diarrhea', 'conjunctivitis', 
]

collected_items = []  # Acumulador global de artículos


class KidsHealthEnSpider(CrawlSpider):
    name = 'kidshealth_en_kids'
    allowed_domains = ['kidshealth.org']

    # Punto de entrada: directorio de niños en inglés
    start_urls = ['https://kidshealth.org/en/parents/infections/']

    rules = (
        Rule(
            LinkExtractor(
                allow_domains=['kidshealth.org'],
                allow=[r'/en/parents/'],
                deny=[
                    r'/en/kids/all-categories',
                    r'/en/kids/word-',
                    r'/search',
                    r'/es/',          # bloquear todo el español
                    r'/en/kids/',
                    r'/en/teens/',
                ],
            ),
            callback='parse_article',
            follow=True,
        ),
    )

    custom_settings = {
        'DEPTH_LIMIT': DEPTH_LIMIT,
        'DEPTH_PRIORITY': 1,
        'CONCURRENT_REQUESTS': CONCURRENT_REQUESTS,
        'CLOSESPIDER_ITEMCOUNT': MAX_ARTICLES,
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 0.5,   # Pequeña pausa entre requests, evita bans
        'USER_AGENT': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0 Safari/537.36'
        ),
        'LOG_LEVEL': 'WARNING',
    }

    @classmethod
    def from_crawler(cls, crawler):
        spider = super().from_crawler(crawler)
        crawler.signals.connect(
            spider.spider_closed,
            signal=scrapy.signals.spider_closed
        )
        return spider

    def spider_closed(self, spider):
        df = pd.DataFrame(collected_items)
        # Eliminar duplicados por URL (por si acaso el spider visita la misma página dos veces)
        df = df.drop_duplicates(subset='url').reset_index(drop=True)

        print(f'\n✅ Scraping terminado: {len(df)} artículos recopilados.\n')
        print(df[['title', 'url', 'num_keywords']].to_string())

        # Guardar en parquet (cómodo para pandas/polars) y también en JSON lines

        parquet_path = DATA_DIR / 'kidshealth_en_parents_infections.parquet'
        jsonl_path = DATA_DIR / 'kidshealth_en_parents_infections.jsonl'
        # Guardar los archivos
        df.to_parquet(parquet_path, index=False)
        df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)
        print(f'\n💾 Guardado en {parquet_path} y {jsonl_path}')

    def parse_article(self, response):
        # Solo procesar páginas HTML
        if not response.url.endswith('.html'):
            return

        # Extraer texto limpio con trafilatura
        text = trafilatura.extract(
            response.body,
            favor_precision=True,
            include_comments=False,
            include_tables=False,
            deduplicate=True,
        )

        # Descartar si no se extrajo texto útil
        if not text or len(text.strip()) < 200:  # mínimo ~200 caracteres
            return

        # Filtro por keywords: al menos una debe aparecer en el texto
        matched = [kw for kw in KEYWORDS if kw in text.lower()]
        if not matched:
            return

        item = {
            'url': response.url,
            'title': response.css('title::text').get('').strip(),
            'matched_keywords': matched,
            'num_keywords': len(matched),
            'text': text,
        }
        collected_items.append(item)
        yield item