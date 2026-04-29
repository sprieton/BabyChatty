# import pandas as pd
# df = pd.read_parquet('kidshealth_en_kids.parquet')
# print(df.shape)           # (nº artículos, nº columnas)
# print(df['url'].head(20))
# print(df.iloc[0]['text']) # Ver texto del primer artículo

# print("==================================================================")

# print(df['text'].apply(len).describe())  # longitud de los textos
# print(df['title'].tolist())              # ver todos los títulos


import pandas as pd
from utils import GenConfig as cfg

# Carga de datos
df = pd.read_parquet(cfg.parquet_file)

### 1. Diagnóstico de Salud de los Datos
print("--- Estructura General ---")
print(f"Total de artículos: {df.shape[0]}")
print(f"Columnas disponibles: {df.columns.tolist()}")
print("\n--- Valores Nulos ---")
print(df.isnull().sum())

### 2. Análisis de Contenido (NLP Básico)
# Calculamos longitud de palabras en lugar de solo caracteres
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

print("\n--- Estadísticas de palabras por artículo ---")
print(df['word_count'].describe())

### 3. Detección de Duplicados
# Es común que Scrapy repita URLs si no se configura bien
duplicates = df.duplicated(subset=['url']).sum()
print(f"\nArtículos con URL duplicada: {duplicates}")

### 4. Análisis Temático (Basado en Títulos)
# Esto te ayudará a ver qué temas predominan (ej. infecciones, nutrición, etc.)
from collections import Counter
import re

all_titles = " ".join(df['title'].astype(str)).lower()
words = re.findall(r'\w+', all_titles)
# Filtramos palabras comunes (stop words) manualmente o con librería
common_words = Counter(words).most_common(15)

print("\n--- Palabras más frecuentes en los títulos ---")
for word, freq in common_words:
    print(f"{word}: {freq}")

### 5. Exportar una muestra para revisión manual
# Siempre es bueno leer 5 artículos al azar para validar el scraping
df.sample(5).to_csv('muestra_revision.csv', index=False)

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Descargamos stop words
nltk.download('stopwords')
stop_words = stopwords.words('english') # O 'spanish' según tu dataset
# Añadimos palabras obvias que no ayudan a distinguir temas
stop_words.extend(['child', 'kids', 'health', 'parent', 'help', 'children', 'may', 'use', 'get', 'doctor', 'doctors', 'people', 'baby', 'also', 'make', 'getting', 'like', 'type', 'teen', 'teens'])

# 1. Vectorización (Convertimos texto a matriz de números)
vectorizer = CountVectorizer(
    max_df=0.95,        # Ignora términos que aparecen en >95% de docs
    min_df=2,           # Ignora términos que aparecen en <2 docs
    stop_words=stop_words
)

data_vectorized = vectorizer.fit_transform(df['text'].astype(str))

# 2. Ajuste del modelo LDA
# Probamos con 5 tópicos iniciales (puedes ajustar este número)
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
lda_model.fit(data_vectorized)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\n✨ Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

display_topics(lda_model, vectorizer.get_feature_names_out(), 10)

# import pyLDAvis
# # Cambiamos la forma de importar el módulo de preparación
# import pyLDAvis.lda_model # Para versiones nuevas de scikit-learn y pyLDAvis

# # ... tu código anterior de LDA ...

# print("Preparando la visualización...")

# # En lugar de pyLDAvis.sklearn.prepare, usa esto:
# panel = pyLDAvis.lda_model.prepare(lda_model, data_vectorized, vectorizer)

# # Guardar
# pyLDAvis.save_html(panel, 'analisis_temas.html')
# print("✅ Archivo 'analisis_temas.html' generado con éxito.")