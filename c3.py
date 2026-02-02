# ============================================
# c3_modificado.py — versión extendida:
# - Permite guardar cada corpus procesado como un diccionario independiente
# - posteriormente cargarlo para búsquedas o análisi
# ============================================


import os
import re
import io
import json
import requests
import networkx as nx
from text2graphapi.src.Cooccurrence import Cooccurrence
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict, Counter
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geco3_client.client import GECO3Client
# --------------------------------------------
# CONFIGURACIÓN BASE (desde variables de entorno o config.json)
# --------------------------------------------
def load_config():
    """
    Carga configuración desde variables de entorno o archivo config.json
    Orden de prioridad: Variables de entorno > config.json > valores por defecto
    """
    config = {}

    # Intentar cargar desde archivo config.json
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"Advertencia: No se pudo cargar config.json: {e}")

    # Variables de entorno tienen prioridad
    config["base_url"] = os.getenv("GECO_BASE_URL", config.get("base_url", "http://www.geco.unam.mx/geco3/"))
    config["anon_user"] = os.getenv("GECO_ANON_USER", config.get("anon_user", None))
    config["anon_pass"] = os.getenv("GECO_ANON_PASS", config.get("anon_pass", None))
    config["app_name"] = os.getenv("GECO_APP_NAME", config.get("app_name", None))
    config["app_password"] = os.getenv("GECO_APP_PASSWORD", config.get("app_password", None))
    config["user_token"] = os.getenv("GECO_USER_TOKEN", config.get("user_token", None))
    config["data_dir"] = os.getenv("DATA_DIR", config.get("data_dir", "data"))

    return config

# Cargar configuración
CONFIG = load_config()


def get_client(token=None, is_encrypted=True):
    """
    Factory function to create an authenticated GECO3 client.

    Args:
        token: User token from GECO SSO (optional). If None, uses anonymous login.
        is_encrypted: Whether the token is XOR encrypted (default True for SSO tokens).

    Returns:
        Authenticated GECO3Client instance.
    """
    client = GECO3Client(
        host=CONFIG["base_url"],
        anon_user=CONFIG["anon_user"],
        anon_pass=CONFIG["anon_pass"],
        app_name=CONFIG["app_name"],
        app_password=CONFIG["app_password"]
    )

    try:
        client.login(token=token, is_token_encrypted=is_encrypted if token else False)
    except Exception as e:
        print(f"Token login failed, falling back to anonymous: {e}")
        client.login()  # Fallback to anonymous

    return client


# Directorios de trabajo
DATA_DIR = CONFIG["data_dir"]
TEXTS_DIR = os.path.join(DATA_DIR, "textos")
LEMAS_DIR = os.path.join(DATA_DIR, "lemas")
GRAPH_DIR = os.path.join(DATA_DIR, "grafos")
os.makedirs(TEXTS_DIR, exist_ok=True)
os.makedirs(LEMAS_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# --------------------------------------------
# CLASES Y FUNCIONES EXISTENTES
# --------------------------------------------

# (aquí van las clases TextProcessor, GraphBuilder y ReverseDict)
# No las cambio, solo las dejamos igual que en tu c3.py original


# ---------------------------
# INICIALIZACIÓN DE MODELOS
# ---------------------------
try:
    nlp = spacy.load("es_core_news_lg")
except:
    print(" Modelo de spaCy no encontrado. Instala con: python -m spacy download es_core_news_md")
    nlp = None

# Stopwords ampliadas
STOPWORDS = set(STOP_WORDS)
STOPWORDS.update([
    'ser', 'estar', 'haber', 'tener', 'hacer', 'poder', 'deber',
    'querer', 'ir', 'ver', 'dar', 'saber', 'decir', 'llegar',
    'pasar', 'poner', 'parecer', 'quedar', 'creer', 'llevar',
    'dejar', 'seguir', 'encontrar', 'llamar', 'venir', 'pensar',
    'salir', 'volver', 'tomar', 'conocer', 'vivir', 'sentir',
    'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete',
    'ocho', 'nueve', 'diez', 'cien', 'mil', 'primero', 'segundo',
    'último', 'mismo', 'otro', 'todo', 'cada', 'mucho', 'poco',
    'más', 'menos', 'muy', 'tan', 'tanto', 'bastante', 'demasiado'
])

# ---------------------------
# FUNCIONES MEJORADAS DE PROCESAMIENTO
# ---------------------------


class TextProcessor:
    def __init__(self):
        # Aseguramos que la instancia tenga acceso al objeto nlp global
        global nlp 
        self.nlp = nlp if 'nlp' in globals() else None
        self.cache = {}
        self.pos_weights = {
            'NOUN': 2.0,     # Sustantivos más importantes
            'VERB': 1.5,     # Verbos importantes
            'ADJ': 1.3,      # Adjetivos relevantes
            'ADV': 0.8,      # Adverbios menos peso
            'ADP': 0.3,      # Preposiciones poco peso
            'DET': 0.1,      # Determinantes mínimo peso
            'PRON': 0.2,     # Pronombres poco peso
            'CONJ': 0.1      # Conjunciones mínimo peso
        }

    def limpiar_texto_avanzado(self, texto):
        """Limpieza más sofisticada del texto."""
        # Eliminar URLs
        texto = re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', texto)
        # Eliminar emails
        texto = re.sub(r'\S+@\S+', '', texto)
        # Eliminar números solos
        texto = re.sub(r'\b\d+\b', '', texto)
        # Mantener solo letras y espacios (con acentos)
        texto = re.sub(r'[^a-záéíóúñü\s]', ' ', texto.lower())
        # Eliminar espacios múltiples
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto

    def lematizar_con_spacy(self, texto):
        """
        Lematización optimizada para textos largos (Chunking).
        Divide el texto en bloques de 100,000 caracteres para evitar desbordamiento de memoria.
        """
        tokens_procesados = []

        # 1. Si NO tenemos spaCy, usamos un método manual simple para evitar fallos de API con textos gigantes
        if not nlp:
            print("Aviso: SpaCy no está cargado. Usando tokenización simple rápida.")
            # Tokenización simple (split) para no saturar la API externa de Freeling con 3MB
            palabras = texto.split()
            for w in palabras:
                if len(w) > 2 and w not in STOPWORDS:
                    tokens_procesados.append({
                        'lema': w, # Sin spaCy, el lema es la palabra misma
                        'pos': 'UNK',
                        'texto': w
                    })
            return tokens_procesados

        # 2. Configurar spaCy para permitir textos más largos (por seguridad)
        nlp.max_length = len(texto) + 50000

        # 3. PROCESAMIENTO POR LOTES (Chunking)
        # Procesamos de 100,000 en 100,000 caracteres para no saturar la RAM
        tamano_lote = 100000 
        
        print(f"   > Procesando texto extenso ({len(texto)} chars) en bloques de {tamano_lote}...")
        
        for i in range(0, len(texto), tamano_lote):
            # Cortar un pedazo del texto
            lote = texto[i : i + tamano_lote]
            
            # Procesar ese pedazo
            doc = nlp(lote)

            for token in doc:
                if not token.is_stop and not token.is_punct and len(token.text) > 2:
                    tokens_procesados.append({
                        'lema': token.lemma_.lower(),
                        'pos': token.pos_,
                        'texto': token.text.lower()
                    })

        return tokens_procesados

    def lematizar_freeling_mejorado(self, texto):
        """Versión mejorada de lematización con FreeLing."""
        if texto in self.cache:
            return self.cache[texto]

        url = "http://www.corpus.unam.mx/servicio-freeling/analyze.php"
        params = {"outf": "tagged", "format": "json"}

        try:
            archivo = io.BytesIO(texto.encode("utf-8"))
            archivo.name = "texto.txt"
            files = {'file': archivo}
            r = requests.post(url, files=files, params=params, timeout=30)
            data = r.json()

            tokens_procesados = []
            for sent in data:
                for w in sent:
                    if len(w["lemma"]) > 2 and w["lemma"].lower() not in STOPWORDS:
                        tokens_procesados.append({
                            'lema': w["lemma"].lower(),
                            'pos': w.get("tag", "UNK")[0] if "tag" in w else "UNK",
                            'texto': w["form"].lower()
                        })

            self.cache[texto] = tokens_procesados
            return tokens_procesados
        except Exception as e:
            print(f"Error en lematización: {e}")
            # Fallback: tokenización simple
            return [{'lema': w, 'pos': 'UNK', 'texto': w}
                    for w in texto.split() if len(w) > 2 and w not in STOPWORDS]

# ---------------------------
# CONSTRUCCIÓN MEJORADA DEL GRAFO
# ---------------------------


class GraphBuilder:
    def __init__(self, processor):
        self.processor = processor
        self.vocab_freq = Counter()
        self.word_contexts = defaultdict(set)

    def construir_grafo_mejorado(self, tokens_procesados, window_size=15):
        """Construcción mejorada del grafo con pesos contextuales."""
        G = nx.Graph()

        # Extraer solo lemas para el grafo
        lemas = [t['lema'] for t in tokens_procesados]

        # Calcular frecuencias
        self.vocab_freq = Counter(lemas)

        # Filtrar palabras muy raras o muy comunes
        total_words = len(lemas)
        lemas_filtrados = []
        for lema in lemas:
            freq = self.vocab_freq[lema] / total_words
            if 0.0001 < freq < 1.0:  # Entre 0.01% y 10% de frecuencia
                lemas_filtrados.append(lema)

        # Construir grafo con ventana deslizante
        for i, palabra_central in enumerate(lemas_filtrados):
            # Ventana de contexto
            inicio = max(0, i - window_size)
            fin = min(len(lemas_filtrados), i + window_size + 1)

            for j in range(inicio, fin):
                if i != j:
                    palabra_contexto = lemas_filtrados[j]
                    distancia = abs(i - j)
                    peso = 1.0 / distancia  # Peso inversamente proporcional a la distancia

                    if G.has_edge(palabra_central, palabra_contexto):
                        G[palabra_central][palabra_contexto]['weight'] += peso
                    else:
                        G.add_edge(palabra_central,
                                   palabra_contexto, weight=peso)

                    # Guardar contextos para cada palabra
                    self.word_contexts[palabra_central].add(palabra_contexto)

        # Calcular métricas adicionales del grafo
        for node in G.nodes():
            G.nodes[node]['frequency'] = self.vocab_freq[node]
            G.nodes[node]['degree'] = G.degree(node)

        return G

    def calcular_embeddings_contextuales(self, G, dim=50):
        """Crear embeddings basados en el grafo usando Node2Vec simplificado."""
        try:
            from node2vec import Node2Vec
            node2vec = Node2Vec(G, dimensions=dim,
                                walk_length=10, num_walks=100, workers=4)
            model = node2vec.fit(window=5, min_count=1, batch_words=4)
            return model.wv
        except ImportError:
            # Fallback: representación basada en vecinos
            embeddings = {}
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                vector = np.zeros(len(G.nodes()))
                for n in neighbors:
                    idx = list(G.nodes()).index(n)
                    vector[idx] = G[node][n]['weight'] if G.has_edge(
                        node, n) else 0
                embeddings[node] = vector
            return embeddings

# ---------------------------
# SISTEMA DE BÚSQUEDA MEJORADO
# ---------------------------


class ReverseDict:
    def __init__(self, grafo, processor, builder):
        self.grafo = grafo
        self.processor = processor
        self.builder = builder
        self.tfidf = None
        self.tfidf_matrix = None
        self.vocab = list(grafo.nodes())
        self._preparar_tfidf()

        # --- CORRECCIÓN: Calculamos la centralidad estática aquí ---
        # Esto inicializa la variable self.centrality_scores que faltaba.
        try:
            # Usamos degree_centrality porque es muy rápido y eficiente para grafos grandes
            # (Betweenness centrality sería mejor, pero tardaría horas en grafos grandes)
            self.centrality_scores = nx.degree_centrality(self.grafo)
        except Exception as e:
            print(f"Advertencia: No se pudo calcular centralidad ({e}). Se usarán valores por defecto.")
            self.centrality_scores = {n: 0.0 for n in self.grafo.nodes()}
        # -----------------------------------------------------------

        # Preparar el resto de componentes (TF-IDF, etc.)
        self._preparar_tfidf()


    def _preparar_tfidf(self):
        """Preparar vectorizador TF-IDF para búsquedas."""
        # 1. Validación de seguridad: Si no hay palabras, salir sin error.
        if not self.vocab:
            self.tfidf = None
            self.tfidf_matrix = None
            return

        # Crear documentos falsos basados en los contextos (vecinos)
        documentos = []
        for palabra in self.vocab:
            contexto = list(self.builder.word_contexts.get(palabra, [palabra]))
            # Si el contexto está vacío, usamos la palabra misma para que no sea string vacío
            texto = " ".join(contexto) if contexto else palabra
            documentos.append(texto)

        try:
            # 2. SOLUCIÓN CLAVE: Cambiamos max_df a 1.0 (100%)
            # Esto evita que elimine palabras incluso si aparecen en todos lados.
            self.tfidf = TfidfVectorizer(
                max_features=1000, 
                min_df=1, 
                max_df=1.0  # <--- CAMBIO IMPORTANTE: Antes era 0.9
            )
            self.tfidf_matrix = self.tfidf.fit_transform(documentos)
            
        except ValueError:
            # Si aún así falla (ej. palabras de 1 letra que scikit borra), no rompemos la app
            print("Advertencia: No se pudo generar matriz TF-IDF (vocabulario insuficiente).")
            self.tfidf = None
            self.tfidf_matrix = None
    # ------------------------------------------------------------------
    # ESTRATEGIAS DE BÚSQUEDA (MÉTODOS AUXILIARES)
    # ------------------------------------------------------------------

    def _estrategia_pagerank(self, tokens):
        """Calcula relevancia usando PageRank personalizado."""
        # Crear vector de personalización
        personalization = {n: 0.0 for n in self.grafo.nodes()}
        found_any = False
        
        for token in tokens:
            if token in personalization:
                personalization[token] = 1.0
                found_any = True
        
        if not found_any:
            return {}

        try:
            # Ejecutar PageRank con salto alto (alpha=0.85) hacia los términos de búsqueda
            scores = nx.pagerank(self.grafo, alpha=0.85, personalization=personalization, weight='weight')
            return scores
        except Exception as e:
            print(f"Error en PageRank: {e}")
            return {}

    def _estrategia_cosine(self, tokens):
        """Calcula similitud coseno basada en co-ocurrencia (TF-IDF)."""
        if self.tfidf is None or self.tfidf_matrix is None:
            return {}

        try:
            query_str = " ".join(tokens)
            query_vec = self.tfidf.transform([query_str])
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            scores = {}
            # Usamos get_feature_names_out() para versiones nuevas de scikit-learn
            # Si usas una versión vieja, podría ser get_feature_names()
            try:
                feature_names = self.tfidf.get_feature_names_out()
            except AttributeError:
                feature_names = self.tfidf.get_feature_names()

            # --- CORRECCIÓN DE SEGURIDAD ---
            # Iteramos solo hasta el límite más pequeño para evitar el error "Index out of bounds"
            limit = min(len(similarities), len(feature_names))
            
            for idx in range(limit):
                score = similarities[idx]
                if score > 0:
                    word = feature_names[idx]
                    if word in self.grafo:
                        scores[word] = float(score)
            # -------------------------------
            
            return scores
            
        except Exception as e:
            print(f"Error recuperable en Coseno: {e}")
            return {}
        
    def _estrategia_spreading(self, tokens):
        """Propagación de activación simple (vecinos directos)."""
        scores = {}
        for token in tokens:
            if token in self.grafo:
                # La palabra misma de la definición recibe puntos
                scores[token] = scores.get(token, 0) + 1.0
                
                # Sus vecinos reciben puntos basados en el peso de la conexión
                for neighbor, attr in self.grafo[token].items():
                    weight = attr.get('weight', 1.0)
                    # Atenuación: el vecino recibe menos que el nodo original
                    scores[neighbor] = scores.get(neighbor, 0) + (weight * 0.5)
        return scores



    def buscar_multiple_estrategias(self, definition, top_k=10):
        # 1. Preprocesar definición
        tokens = self.processor.lematizar_con_spacy(definition)
        filtered_tokens = [t['lema'] for t in tokens] # Usamos solo los lemas
        
        if not filtered_tokens:
            return []

        # 2. Calcular scores de cada estrategia
        scores_pagerank = self._estrategia_pagerank(filtered_tokens)
        scores_cosine = self._estrategia_cosine(filtered_tokens)
        scores_spreading = self._estrategia_spreading(filtered_tokens)
        
        # Pasamos la 'definition' ORIGINAL (string completo) a la estrategia semántica
        scores_semantic = self._estrategia_semantica(definition)

        # Centralidad (ya la tienes precalculada, es estática)
        scores_centrality = self.centrality_scores

        # 3. Combinar (Normalización y Pesos Ajustados)
        # RECOMENDACIÓN: Dar más peso a la semántica y al coseno (contexto)
        weights = {
            "pagerank": 0.20,   # Antes 0.35
            "cosine": 0.30,     # Antes 0.35 (Contexto local)
            "spreading": 0.10,  # Antes 0.20
            "centrality": 0.05, # Antes 0.10 (Importancia global)
            "semantic": 0.35    # NUEVO: Significado profundo
        }

        final_scores = {}
        all_nodes = set(self.grafo.nodes())

        # Normalizar scores (0 a 1) para poder sumarlos
        def normalize(score_dict):
            if not score_dict: return {}
            max_val = max(score_dict.values())
            if max_val == 0: return score_dict
            return {k: v / max_val for k, v in score_dict.items()}

        norm_pagerank = normalize(scores_pagerank)
        norm_cosine = normalize(scores_cosine)
        norm_spreading = normalize(scores_spreading)
        norm_centrality = normalize(scores_centrality)
        norm_semantic = normalize(scores_semantic) # Normalizar la semántica

        for node in all_nodes:
            s_pr = norm_pagerank.get(node, 0)
            s_cos = norm_cosine.get(node, 0)
            s_spr = norm_spreading.get(node, 0)
            s_cen = norm_centrality.get(node, 0)
            s_sem = norm_semantic.get(node, 0) # Score semántico

            # Fórmula maestra
            final_scores[node] = (
                weights["pagerank"] * s_pr +
                weights["cosine"] * s_cos +
                weights["spreading"] * s_spr +
                weights["centrality"] * s_cen +
                weights["semantic"] * s_sem
            )

        # 4. Ordenar y devolver Top-K
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _pagerank_personalizado(self, lemas_def):
        """PageRank con personalización basada en la definición."""
        personalization = {node: 0 for node in self.grafo.nodes()}
        for lema in lemas_def:
            personalization[lema] = 1.0 / len(lemas_def)

        try:
            scores = nx.pagerank(self.grafo, alpha=0.85, personalization=personalization,
                                 max_iter=200, weight='weight')
        except:
            scores = nx.pagerank(self.grafo, alpha=0.85,
                                 max_iter=200, weight='weight')

        return scores

    def _similitud_tfidf(self, definicion):
        """Similitud basada en TF-IDF."""
        if self.tfidf is None:
            return {}

        def_vector = self.tfidf.transform([definicion])
        similitudes = cosine_similarity(
            def_vector, self.tfidf_matrix).flatten()

        scores = {}
        for i, palabra in enumerate(self.vocab):
            scores[palabra] = similitudes[i]

        return scores

    def _propagacion_activacion(self, lemas_def, iteraciones=3):
        """Propagación de activación en el grafo."""
        activacion = {node: 0 for node in self.grafo.nodes()}

        # Activación inicial
        for lema in lemas_def:
            activacion[lema] = 1.0

        # Propagar activación
        for _ in range(iteraciones):
            nueva_activacion = {}
            for node in self.grafo.nodes():
                suma = activacion[node] * 0.5  # Factor de decay
                for vecino in self.grafo.neighbors(node):
                    peso = self.grafo[node][vecino]['weight']
                    suma += activacion[vecino] * peso * 0.1
                nueva_activacion[node] = suma
            activacion = nueva_activacion

        # Normalizar scores
        max_act = max(activacion.values()) if activacion else 1
        return {k: v/max_act for k, v in activacion.items()}

    def _betweenness_local(self, lemas_def, profundidad=2):
        """Centralidad de intermediación en subgrafo local."""
        # Obtener subgrafo local
        nodos_subgrafo = set(lemas_def)
        for lema in lemas_def:
            for _ in range(profundidad):
                vecinos = list(self.grafo.neighbors(lema))
                nodos_subgrafo.update(vecinos[:20])  # Limitar vecinos

        if len(nodos_subgrafo) < 3:
            return {}

        subgrafo = self.grafo.subgraph(nodos_subgrafo)

        try:
            scores = nx.betweenness_centrality(subgrafo, weight='weight',
                                               normalized=True, k=min(10, len(nodos_subgrafo)))
        except:
            scores = {}

        return scores

    def buscar_con_feedback(self, definicion, top_k=10):
        """Búsqueda con opción de retroalimentación."""
        resultados = self.buscar_multiple_estrategias(
            definicion, top_k=top_k*2)

        print("\n Resultados principales:")
        for i, (palabra, score) in enumerate(resultados[:top_k], 1):
            contexto_sample = list(
                self.builder.word_contexts.get(palabra, []))[:3]
            print(f"{i}. {palabra} (score: {score:.4f})")
            if contexto_sample:
                print(f"   Contexto: {', '.join(contexto_sample)}")

        # Retroalimentación opcional
        feedback = input(
            "\n¿Alguna palabra es correcta? (número o 'no'): ").strip()
        if feedback.isdigit():
            idx = int(feedback) - 1
            if 0 <= idx < len(resultados):
                palabra_correcta = resultados[idx][0]
                # Refinar búsqueda usando la palabra correcta
                return self._refinar_busqueda(definicion, palabra_correcta, top_k)

        return [r[0] for r in resultados[:top_k]]

    def _refinar_busqueda(self, definicion, palabra_correcta, top_k):
        """Refinar búsqueda basándose en retroalimentación."""
        print(f"\n Refinando búsqueda con '{palabra_correcta}'...")

        # Obtener contexto de la palabra correcta
        contexto_correcto = self.builder.word_contexts.get(
            palabra_correcta, set())

        # Buscar palabras similares en contexto
        scores = {}
        for palabra in self.vocab:
            if palabra != palabra_correcta:
                contexto_palabra = self.builder.word_contexts.get(
                    palabra, set())
                similitud = len(contexto_correcto & contexto_palabra) / \
                    (len(contexto_correcto | contexto_palabra) + 1)
                scores[palabra] = similitud

        resultados = sorted(
            scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [r[0] for r in resultados]
    

    def _estrategia_semantica(self, definition_text):
        """
        Calcula similitud semántica usando la frase completa (con stopwords).
        Esto permite capturar el contexto de verbos como 'hacer' o 'ser'.
        """
        scores = {}
        
        # Usamos el texto original, NO los tokens filtrados
        if not self.processor.nlp:
            return scores
            
        doc_def = self.processor.nlp(definition_text)
        
        if not doc_def.vector_norm:
            return scores

        # Comparamos contra los nodos del grafo
        for node in self.grafo.nodes():
            # Optimizacion: Si tienes muchos nodos, esto puede ser lento.
            # En el futuro podríamos pre-calcular vectores de nodos.
            doc_node = self.processor.nlp(node)
            
            if doc_node.vector_norm:
                similitud = doc_def.similarity(doc_node)
                if similitud > 0.35: # Umbral un poco más alto
                    scores[node] = similitud
                    
        return scores
    
# ---------------------------------------------------------------
# FUNCIONES DEL CORPUS (actualizadas con filtrado por metadatos)
# ---------------------------------------------------------------


def listar_corpus(client, include_private=False):
    """
    Lista los corpus disponibles desde la API usando GECO3Client.

    Args:
        client: Authenticated GECO3Client instance.
        include_private: If True, also fetches private/collaborative corpora.

    Returns:
        List of corpus dictionaries.
    """
    # Si hay app token, usar corpus de la app; si no, usar corpus públicos
    if client.is_app_logged():
        corpus_list = client.corpus_app()
    else:
        corpus_list = client.corpus_publicos()

    # Include private corpora if requested and user is authenticated
    if include_private:
        try:
            private_corpora = client.corpus_privados()
            existing_ids = {c["id"] for c in corpus_list}
            for c in private_corpora:
                if c["id"] not in existing_ids:
                    corpus_list.append(c)
        except Exception:
            pass  # Ignore if private corpora fetch fails

    print("\nCorpus disponibles:\n")
    for i, c in enumerate(corpus_list, 1):
        print(f"{i}. {c['nombre']} (ID: {c['id']})")
    return corpus_list


def elegir_corpus(corpus_list):
    """Permite elegir un corpus de la lista mostrada."""
    idx = int(input("\nElige un corpus: "))
    return corpus_list[idx - 1]


def listar_documentos(client, corpus_id):
    """
    Lista documentos dentro de un corpus usando GECO3Client.

    Args:
        client: Authenticated GECO3Client instance.
        corpus_id: ID of the corpus.

    Returns:
        List of document dictionaries.
    """
    documentos = client.docs_corpus(corpus_id)
    print("\nDocumentos disponibles:\n")
    for i, d in enumerate(documentos, 1):
        print(f"{i}. {d['archivo']} (ID: {d['id']})")
    return documentos


def elegir_documentos(documentos):
    """Permite seleccionar documentos por número."""
    indices = input("\nElige documentos (ej: 1,3,5): ")
    indices = [int(x.strip()) - 1 for x in indices.split(",")]
    return [documentos[i] for i in indices]


def descargar_documento(client, corpus_id, doc_id):
    """
    Descarga un documento específico por ID usando GECO3Client.

    Args:
        client: Authenticated GECO3Client instance.
        corpus_id: ID of the corpus.
        doc_id: ID of the document.

    Returns:
        Document content as string.
    """
    return client.doc_content(corpus_id, doc_id)


# =====================================
# NUEVA FUNCIÓN: FILTRAR POR METADATOS
# =====================================
def filtrar_documentos_por_metadatos(client, corpus_id):
    """
    Descarga los metadatos del corpus usando GECO3Client.docs_tabla()
    y permite filtrar los documentos por un valor de metadato.

    Args:
        client: Authenticated GECO3Client instance.
        corpus_id: ID of the corpus.
    """
    try:
        # Obtener documentos con metadatos usando GECO3Client
        docs = client.docs_tabla(corpus_id)
    except Exception as e:
        print(f"Error al obtener metadatos del corpus: {e}")
        return []

    if not docs:
        print("No hay documentos disponibles en este corpus.\n")
        return []

    # Obtener todos los nombres de metadatos disponibles
    metadatos_disponibles = set()
    for doc in docs:
        metadatos_disponibles.update(doc.get("metadata", {}).keys())

    if not metadatos_disponibles:
        print("No hay metadatos disponibles para este corpus.\n")
        return [{"id": doc["id"], "archivo": doc["name"]} for doc in docs]

    # Mostrar metadatos disponibles
    metadatos_lista = sorted(metadatos_disponibles)
    print("\nMetadatos disponibles para filtrar:")
    for i, meta_nombre in enumerate(metadatos_lista, 1):
        print(f"{i}. {meta_nombre}")

    opcion = input("\n¿Deseas filtrar por algún metadato? (s/n): ").lower()
    if opcion != "s":
        print("No se aplicará ningún filtro.\n")
        return [{"id": doc["id"], "archivo": doc["name"]} for doc in docs]

    # Elegir metadato
    idx = int(input("\nSelecciona el número del metadato para filtrar: ")) - 1
    meta_nombre = metadatos_lista[idx]

    # Obtener todos los valores disponibles para ese metadato
    valores_disponibles = set()
    for doc in docs:
        valor = doc.get("metadata", {}).get(meta_nombre)
        if valor is not None:
            valores_disponibles.add(valor)
    valores_disponibles = sorted(valores_disponibles)

    if not valores_disponibles:
        print(f"No hay valores registrados para el metadato '{meta_nombre}'.")
        return []

    print(f"\nValores disponibles para '{meta_nombre}':")
    for i, v in enumerate(valores_disponibles, 1):
        print(f"{i}. {v if v else '(vacío)'}")

    vidx = int(input(f"\nElige un valor para '{meta_nombre}': ")) - 1
    valor_elegido = valores_disponibles[vidx]
    print(f"\nFiltrando documentos donde '{meta_nombre}' = '{valor_elegido}'...\n")

    # Filtrar los documentos
    documentos_filtrados = []
    for doc in docs:
        if doc.get("metadata", {}).get(meta_nombre) == valor_elegido:
            documentos_filtrados.append({"id": doc["id"], "archivo": doc["name"]})

    if not documentos_filtrados:
        print("No se encontraron documentos con ese filtro.\n")
        return []

    print(f"\n{len(documentos_filtrados)} documentos encontrados:\n")
    for i, d in enumerate(documentos_filtrados, 1):
        print(f"{i}. {d['archivo']} (ID: {d['id']})")

    return documentos_filtrados


def filtrar_documentos_por_metadatos_api(client, corpus_id, meta_nombre, valor):
    """
    Versión no interactiva para Flask.
    Devuelve lista de documentos que cumplen el filtro (sin pedir input()).
    Usa GECO3Client.docs_tabla() para obtener datos.

    Args:
        client: Authenticated GECO3Client instance.
        corpus_id: ID of the corpus.
        meta_nombre: Name of the metadata field to filter by.
        valor: Value to match.

    Returns:
        List of document dictionaries matching the filter.
    """
    try:
        # Obtener documentos con metadatos usando GECO3Client
        docs = client.docs_tabla(corpus_id)
    except Exception as e:
        print(f"Error al obtener documentos: {e}")
        return []

    # Filtrar documentos por metadato y valor
    documentos_filtrados = []
    for doc in docs:
        doc_valor = doc.get("metadata", {}).get(meta_nombre)
        if doc_valor and doc_valor.strip().lower() == valor.strip().lower():
            documentos_filtrados.append({"id": doc["id"], "archivo": doc["name"]})

    return documentos_filtrados

# =====================================
# NUEVA FUNCIÓN: FILTRAR POR VARIOS METADATOS (para API Flask/app.py)
# =====================================


def filtrar_documentos_por_varios_metadatos_api(client, corpus_id, metas, valores):
    """
    Filtra documentos que cumplan simultáneamente varios metadatos y valores.
    Ejemplo:
        metas = ["Área", "Lengua"]
        valores = ["Medicina", "Español"]
    Devuelve una lista de documentos (diccionarios con id y archivo).
    Usa GECO3Client.docs_tabla() para obtener datos.

    Args:
        client: Authenticated GECO3Client instance.
        corpus_id: ID of the corpus.
        metas: List of metadata field names.
        valores: List of values to match (parallel to metas).

    Returns:
        List of document dictionaries matching all filters.
    """
    try:
        # Obtener documentos con metadatos usando GECO3Client
        docs = client.docs_tabla(corpus_id)
    except Exception as e:
        print(f"Error al obtener documentos: {e}")
        return []

    # Crear lista de filtros (nombre_metadato, valor_esperado)
    filtros = list(zip(metas, valores))

    # Filtrar documentos que cumplan TODOS los criterios
    documentos_filtrados = []
    for doc in docs:
        metadata = doc.get("metadata", {})
        # Verificar que todos los pares (metadato, valor) coincidan
        cumple_todos = all(
            metadata.get(meta_nombre) == valor
            for meta_nombre, valor in filtros
        )
        if cumple_todos:
            documentos_filtrados.append({"id": doc["id"], "archivo": doc["name"]})

    return documentos_filtrados


def obtener_metadatos_corpus(client, corpus_id):
    """
    Obtiene los metadatos disponibles en un corpus y sus valores únicos.

    Args:
        client: Authenticated GECO3Client instance.
        corpus_id: ID of the corpus.

    Returns:
        Dictionary mapping metadata names to lists of unique values.
    """
    docs = client.docs_tabla(corpus_id)

    metadatos = {}
    for doc in docs:
        for key, value in doc.get("metadata", {}).items():
            if key not in metadatos:
                metadatos[key] = set()
            if value:
                metadatos[key].add(value)

    # Convert sets to sorted lists
    return {k: sorted(list(v)) for k, v in metadatos.items()}


# --------------------------------------------
# NUEVAS FUNCIONES PARA DICCIONARIOS
# --------------------------------------------


# --- EN c3.py ---

def guardar_diccionario(nombre_diccionario, grafo, builder, owner="Anónimo"):
    """
    Guarda el grafo y registra al propietario (owner).
    """
    base_name = nombre_diccionario.replace(" ", "_")
    archivo_json = f"{base_name}.json"
    archivo_graphml = f"{base_name}.graphml"

    ruta_json = os.path.join(GRAPH_DIR, archivo_json)
    ruta_graphml = os.path.join(GRAPH_DIR, archivo_graphml)

    # Datos a guardar
    data = {
        "nombre": nombre_diccionario,
        "nodes": list(grafo.nodes(data=True)),
        "edges": list(grafo.edges(data=True)),
        "word_contexts": {k: list(v) for k, v in builder.word_contexts.items()},
        "vocab_freq": dict(builder.vocab_freq)
    }

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    try:
        nx.write_graphml(grafo, ruta_graphml)
    except Exception as e:
        print(f"No se pudo guardar en formato GraphML: {e}")

    # ----- Actualizar índice global -----
    index_path = os.path.join(GRAPH_DIR, "diccionarios_index.json")
    index = []
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

    # Eliminar entrada anterior si existe (sobrescribir)
    index = [d for d in index if d["nombre"] != nombre_diccionario]

    # Agregar nueva entrada con el owner
    index.append({
        "nombre": nombre_diccionario,
        "archivo_json": archivo_json,
        "archivo_graphml": archivo_graphml,
        "owner": owner  # <--- NUEVO: Guardamos quién lo creó
    })

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Diccionario '{nombre_diccionario}' guardado por '{owner}'.")


def borrar_diccionario(nombre_diccionario, usuario_solicitante):
    """
    Borra un diccionario solo si el usuario_solicitante es el dueño.
    Retorna (Exito:bool, Mensaje:str).
    """
    index_path = os.path.join(GRAPH_DIR, "diccionarios_index.json")
    if not os.path.exists(index_path):
        return False, "No existen diccionarios."

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    # Buscar el diccionario en el índice
    target = next((d for d in index if d["nombre"] == nombre_diccionario), None)
    
    if not target:
        return False, "Diccionario no encontrado."

    # VALIDACIÓN DE PROPIEDAD
    # Si el diccionario tiene owner, verificamos que coincida.
    # Si no tiene owner (diccionarios viejos), permitimos borrar o restringimos (aquí restringimos).
    owner_real = target.get("owner")
    
    # Nota: Si usuario_solicitante es None o vacío, asumimos "Anónimo" o bloqueamos.
    if owner_real and owner_real != usuario_solicitante:
        return False, f"Acceso denegado. Este diccionario pertenece a '{owner_real}'."

    # Borrar archivos físicos
    try:
        if "archivo_json" in target:
            ruta = os.path.join(GRAPH_DIR, target["archivo_json"])
            if os.path.exists(ruta):
                os.remove(ruta)
        
        if "archivo_graphml" in target:
            ruta = os.path.join(GRAPH_DIR, target["archivo_graphml"])
            if os.path.exists(ruta):
                os.remove(ruta)
            
    except Exception as e:
        return False, f"Error borrando archivos físicos: {e}"

    # Actualizar y guardar índice sin el diccionario borrado
    index = [d for d in index if d["nombre"] != nombre_diccionario]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return True, f"Diccionario '{nombre_diccionario}' eliminado correctamente."

def cargar_diccionario(nombre_diccionario):
    """Carga un diccionario guardado desde disco (formato JSON)."""
    index_path = os.path.join(GRAPH_DIR, "diccionarios_index.json")
    if not os.path.exists(index_path):
        print("No hay diccionarios guardados aún.")
        return None, None, None

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    dic_entry = next(
        (d for d in index if d["nombre"] == nombre_diccionario), None)
    if not dic_entry:
        print(f"No se encontró el diccionario '{nombre_diccionario}'.")
        return None, None, None

    # Nuevo formato: preferir archivo_json, mantener compatibilidad con versiones viejas
    ruta = None
    if "archivo_json" in dic_entry:
        ruta = os.path.join(GRAPH_DIR, dic_entry["archivo_json"])
    elif "archivo" in dic_entry:
        ruta = os.path.join(GRAPH_DIR, dic_entry["archivo"])
    else:
        print(
            f"El registro del diccionario '{nombre_diccionario}' no tiene archivo asociado.")
        return None, None, None

    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruir grafo
    G = nx.Graph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from(data["edges"])

    # Restaurar builder y processor
    processor = TextProcessor()
    builder = GraphBuilder(processor)
    builder.word_contexts = {
        k: set(v) for k, v in data.get("word_contexts", {}).items()}
    builder.vocab_freq = Counter(data.get("vocab_freq", {}))

    print(
        f"Diccionario '{nombre_diccionario}' cargado correctamente desde JSON.")
    print(f"   Nodos: {len(G.nodes())}, Aristas: {len(G.edges())}")

    return G, processor, builder


# ---------------------------
# FUNCIONES DE EVALUACIÓN
# ---------------------------


def evaluar_sistema(diccionario_inverso, pruebas):
    """Evaluar el sistema con casos de prueba."""
    print("\n" + "="*50)
    print("EVALUACIÓN DEL SISTEMA")
    print("="*50)

    for definicion, esperado in pruebas:
        print(f"\nDefinición: '{definicion}'")
        print(f"Esperado: {esperado}")
        resultados = diccionario_inverso.buscar_multiple_estrategias(
            definicion, top_k=10)
        resultados_palabras = [r[0] for r in resultados[:5]]
        print(f"Obtenido: {resultados_palabras}")
        if esperado in resultados_palabras:
            print(" CORRECTO")
        else:
            print(" INCORRECTO")


# --------------------------------------------
# PROCESAMIENTO PRINCIPAL (modo terminal)
# --------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("        DICCIONARIO INVERSO — MODO TERMINAL")
    print("=" * 60)

    # Create client for terminal mode (uses config token or anonymous)
    token = CONFIG.get("user_token")
    client = get_client(token=token, is_encrypted=False)

    # Mostrar corpus disponibles
    corpus_list = listar_corpus(client)
    corpus = elegir_corpus(corpus_list)
    corpus_id = corpus.get("id")

    # Mostrar documentos
    documentos = filtrar_documentos_por_metadatos(client, corpus_id)
    docs_sel = elegir_documentos(documentos)

    print("\nDescargando y procesando documentos seleccionados...\n")

    textos = []
    processor = TextProcessor()
    builder = GraphBuilder(processor)

    for d in docs_sel:
        try:
            txt = descargar_documento(client, corpus_id, d.get("id"))
            texto_limpio = processor.limpiar_texto_avanzado(txt)
            textos.append(texto_limpio)
            print(f"  ✓ {d['archivo']} procesado.")
        except Exception as e:
            print(f"Error al procesar {d['archivo']}: {e}")

    if not textos:
        print("No se procesó ningún texto. Saliendo.")
        exit()

    # Unir textos y procesar
    texto_unido = " ".join(textos)
    print("\nLematizando texto...")
    tokens_procesados = processor.lematizar_con_spacy(texto_unido)
    print(f"  {len(tokens_procesados)} tokens procesados.")

    print("\nConstruyendo grafo...")
    grafo = builder.construir_grafo_mejorado(tokens_procesados)
    print(
        f"  Grafo creado con {len(grafo.nodes())} nodos y {len(grafo.edges())} aristas.")

    # Guardar como diccionario
    nombre_dic = input("\nIntroduce un nombre para este diccionario: ").strip()
    guardar_diccionario(nombre_dic, grafo, builder)

    # Opcional: cargar un diccionario existente
    index_path = os.path.join(GRAPH_DIR, "diccionarios_index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        if index:
            print("\nDiccionarios disponibles:")
            for i, d in enumerate(index, 1):
                print(f"{i}. {d['nombre']}")
            opcion = input(
                "\n¿Deseas cargar uno existente para prueba? (s/n): ").lower()
            if opcion == "s":
                idx = int(input("Elige el número del diccionario: ")) - 1
                nombre_sel = index[idx]["nombre"]
                grafo, processor, builder = cargar_diccionario(nombre_sel)
                if grafo:
                    print(
                        f"Diccionario '{nombre_sel}' cargado correctamente.")
                    print(
                        f"   Nodos: {len(grafo.nodes())}, Aristas: {len(grafo.edges())}")
                else:
                    print("No se pudo cargar el diccionario.")

    print("\n Proceso completado.")

    # 5. Inicializar sistema de búsqueda
    print("\n Inicializando sistema de búsqueda...")
    diccionario_inverso = ReverseDict(grafo, processor, builder)

    # 6. Casos de prueba (opcional)
    ejecutar_pruebas = input("\n¿Ejecutar casos de prueba? (s/n): ").lower()
    if ejecutar_pruebas == 's':
        pruebas = [
            ("animal doméstico que ladra", "perro"),
            ("lugar donde se guardan libros", "biblioteca"),
            ("persona que enseña en la escuela", "profesor"),
            ("líquido transparente para beber", "agua"),
            ("vehículo de dos ruedas", "bicicleta")
        ]
        evaluar_sistema(diccionario_inverso, pruebas)

    # 7. Búsqueda interactiva
    print("\n" + "="*50)
    print("MODO DE BÚSQUEDA INTERACTIVA")
    print("="*50)
    print("Comandos especiales:")
    print("  • 'salir' - terminar el programa")
    print("  • 'ayuda' - mostrar ayuda")
    print("  • 'stats' - mostrar estadísticas del grafo")

    while True:
        definicion = input("\n Introduce una definición: ").strip()

        if definicion.lower() == "salir":
            print(" ¡Hasta luego!")
            break
        elif definicion.lower() == "ayuda":
            print("\n AYUDA:")
            print("  • Escribe definiciones descriptivas")
            print("  • Usa palabras clave relacionadas")
            print("  • Evita palabras muy comunes")
            print("  • Puedes dar retroalimentación para mejorar resultados")
        elif definicion.lower() == "stats":
            print(f"\n Estadísticas del sistema:")
            print(f"  • Nodos en grafo: {len(grafo.nodes())}")
            print(f"  • Aristas en grafo: {len(grafo.edges())}")
            print(
                f"  • Palabras más frecuentes: {builder.vocab_freq.most_common(5)}")
        else:
            # Búsqueda con retroalimentación
            resultados = diccionario_inverso.buscar_con_feedback(
                definicion, top_k=10)

            # Opción de búsqueda alternativa
            if input("\n¿Probar búsqueda alternativa? (s/n): ").lower() == 's':
                resultados_alt = diccionario_inverso.buscar_multiple_estrategias(
                    definicion, top_k=15)
                print("\n Resultados alternativos:")
                for i, (palabra, score) in enumerate(resultados_alt[:10], 1):
                    print(f"{i}. {palabra} (score: {score:.4f})")

    print("\n Programa terminado correctamente.")