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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------
# CONFIGURACIÓN BASE
# --------------------------------------------
BASE_URL = "http://www.geco.unam.mx/geco3/proyectos/apidocs/corpus/"
headers = {
    "authorization": "Token b882293a53d01d42cdb4f06c80f56c96e70fffef"
}

# Directorios de trabajo
TEXTS_DIR = "data/textos"
LEMAS_DIR = "data/lemas"
GRAPH_DIR = "data/grafos"
os.makedirs(TEXTS_DIR, exist_ok=True)
os.makedirs(LEMAS_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# --------------------------------------------
# CLASES Y FUNCIONES EXISTENTES
# --------------------------------------------

# (aquí van tus clases TextProcessor, GraphBuilder y ReverseDict)
# No las cambio, solo las dejamos igual que en tu c3.py original


# ---------------------------
# INICIALIZACIÓN DE MODELOS
# ---------------------------
try:
    nlp = spacy.load("es_core_news_md")
except:
    print(" Modelo de spaCy no encontrado. Instala con: python -m spacy download es_core_news_md")
    nlp = None

# Stopwords ampliadas
STOPWORDS = set(stopwords.words("spanish"))
STOPWORDS_ADICIONALES = {
    'ser', 'estar', 'haber', 'tener', 'hacer', 'poder', 'deber',
    'querer', 'ir', 'ver', 'dar', 'saber', 'decir', 'llegar',
    'pasar', 'poner', 'parecer', 'quedar', 'creer', 'llevar',
    'dejar', 'seguir', 'encontrar', 'llamar', 'venir', 'pensar',
    'salir', 'volver', 'tomar', 'conocer', 'vivir', 'sentir',
    'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete',
    'ocho', 'nueve', 'diez', 'cien', 'mil', 'primero', 'segundo',
    'último', 'mismo', 'otro', 'todo', 'cada', 'mucho', 'poco',
    'más', 'menos', 'muy', 'tan', 'tanto', 'bastante', 'demasiado'
}
STOPWORDS.update(STOPWORDS_ADICIONALES)

# ---------------------------
# FUNCIONES MEJORADAS DE PROCESAMIENTO
# ---------------------------


class TextProcessor:
    def __init__(self):
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
        """Lematización mejorada con spaCy si está disponible."""
        if not nlp:
            return self.lematizar_freeling_mejorado(texto)

        doc = nlp(texto)
        tokens_procesados = []

        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 2:
                # Guardar lema con su POS tag para uso posterior
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

    def _preparar_tfidf(self):
        """Preparar vectorizador TF-IDF para búsquedas."""
        # Crear documentos para cada palabra usando sus contextos
        documentos = []
        for palabra in self.vocab:
            contexto = list(self.builder.word_contexts.get(palabra, [palabra]))
            documentos.append(" ".join(contexto))

        self.tfidf = TfidfVectorizer(max_features=1000, min_df=1, max_df=0.9)
        self.tfidf_matrix = self.tfidf.fit_transform(documentos)

    def buscar_multiple_estrategias(self, definicion, top_k=15):
        """Búsqueda combinando múltiples estrategias."""
        # Procesar definición
        definicion_limpia = self.processor.limpiar_texto_avanzado(definicion)
        tokens_def = self.processor.lematizar_con_spacy(definicion_limpia) if nlp else \
            self.processor.lematizar_freeling_mejorado(definicion_limpia)

        lemas_def = [t['lema']
                     for t in tokens_def if t['lema'] in self.grafo.nodes()]

        if not lemas_def:
            print(" No se encontraron palabras de la definición en el corpus.")
            return []

        # Estrategia 1: PageRank personalizado
        scores_pr = self._pagerank_personalizado(lemas_def)

        # Estrategia 2: Similitud TF-IDF
        scores_tfidf = self._similitud_tfidf(definicion_limpia)

        # Estrategia 3: Propagación de activación
        scores_prop = self._propagacion_activacion(lemas_def)

        # Estrategia 4: Centralidad de intermediación local
        scores_bet = self._betweenness_local(lemas_def)

        # Combinar scores con pesos
        scores_combinados = defaultdict(float)
        pesos = {
            'pagerank': 0.35,
            'tfidf': 0.30,
            'propagacion': 0.25,
            'betweenness': 0.10
        }

        for palabra in self.vocab:
            if palabra not in lemas_def:  # No incluir palabras de la definición
                score = 0
                score += scores_pr.get(palabra, 0) * pesos['pagerank']
                score += scores_tfidf.get(palabra, 0) * pesos['tfidf']
                score += scores_prop.get(palabra, 0) * pesos['propagacion']
                score += scores_bet.get(palabra, 0) * pesos['betweenness']
                scores_combinados[palabra] = score

        # Ordenar y retornar top-k
        resultados = sorted(scores_combinados.items(),
                            key=lambda x: x[1], reverse=True)
        return [(palabra, score) for palabra, score in resultados[:top_k]]

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


# ---------------------------------------------------------------
# FUNCIONES DEL CORPUS (actualizadas con filtrado por metadatos)
# ---------------------------------------------------------------


def listar_corpus():
    """Lista los corpus disponibles desde la API."""
    r = requests.get(BASE_URL, headers=headers)
    r.raise_for_status()
    corpus_list = r.json()["data"]["proyectos"]
    print("\nCorpus disponibles:\n")
    for i, c in enumerate(corpus_list, 1):
        print(f"{i}. {c['nombre']} (ID: {c['id']})")
    return corpus_list


def elegir_corpus(corpus_list):
    """Permite elegir un corpus de la lista mostrada."""
    idx = int(input("\nElige un corpus: "))
    return corpus_list[idx - 1]


def listar_documentos(corpus_id):
    """Lista documentos dentro de un corpus."""
    r = requests.get(BASE_URL + str(corpus_id), headers=headers)
    r.raise_for_status()
    documentos = r.json()["data"]
    print("\nDocumentos disponibles:\n")
    for i, d in enumerate(documentos, 1):
        print(f"{i}. {d['archivo']} (ID: {d['id']})")
    return documentos


def elegir_documentos(documentos):
    """Permite seleccionar documentos por número."""
    indices = input("\nElige documentos (ej: 1,3,5): ")
    indices = [int(x.strip()) - 1 for x in indices.split(",")]
    return [documentos[i] for i in indices]


def descargar_documento(corpus_id, doc_id):
    """Descarga un documento específico por ID."""
    url = f"{BASE_URL}{corpus_id}/{doc_id}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()
    return data.get("data", "")


# =====================================
# NUEVA FUNCIÓN: FILTRAR POR METADATOS
# =====================================
def filtrar_documentos_por_metadatos(corpus_id):
    """
    Descarga los metadatos del corpus (endpoint /tabla)
    y permite filtrar los documentos por un valor de metadato.
    """
    url = f"{BASE_URL}{corpus_id}/tabla"
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()["data"]
    except Exception as e:
        print(f"Error al obtener metadatos del corpus: {e}")
        return []

    metadatos = data.get("metadatos", [])
    tabla = data.get("tabla", [])

    # Crear índice de metadatos y valores posibles
    meta_dict = {}
    for m in metadatos:
        meta_id, meta_nombre, valores = m
        meta_dict[meta_id] = {"nombre": meta_nombre, "valores": set(valores)}

    if not meta_dict:
        print("No hay metadatos disponibles para este corpus.\n")
        return [{"id": t[0], "archivo": t[1]} for t in tabla]

    # Mostrar tipos de metadatos
    print("\nMetadatos disponibles para filtrar:")
    for i, m in enumerate(meta_dict.values(), 1):
        print(f"{i}. {m['nombre']}")

    opcion = input("\n¿Deseas filtrar por algún metadato? (s/n): ").lower()
    if opcion != "s":
        print("No se aplicará ningún filtro.\n")
        # Convertimos 'tabla' a lista de diccionarios (id, archivo)
        return [{"id": t[0], "archivo": t[1]} for t in tabla]

    # Elegir metadato
    idx = int(input("\nSelecciona el número del metadato para filtrar: ")) - 1
    meta_id = list(meta_dict.keys())[idx]
    meta_nombre = meta_dict[meta_id]["nombre"]

    # Obtener todos los valores disponibles de ese metadato
    valores_disponibles = set()
    for doc in tabla:
        for mid, valor in doc[2]:
            if mid == meta_id:
                valores_disponibles.add(valor)
    valores_disponibles = sorted(valores_disponibles)

    if not valores_disponibles:
        print(
            f"No hay valores registrados para el metadato '{meta_nombre}'.")
        return []

    print(f"\nValores disponibles para '{meta_nombre}':")
    for i, v in enumerate(valores_disponibles, 1):
        print(f"{i}. {v if v else '(vacío)'}")

    vidx = int(input(f"\nElige un valor para '{meta_nombre}': ")) - 1
    valor_elegido = valores_disponibles[vidx]
    print(
        f"\nFiltrando documentos donde '{meta_nombre}' = '{valor_elegido}'...\n")

    # Filtrar los documentos
    documentos_filtrados = []
    for doc in tabla:
        doc_id, archivo, metas = doc
        for mid, valor in metas:
            if mid == meta_id and valor == valor_elegido:
                documentos_filtrados.append({"id": doc_id, "archivo": archivo})
                break

    if not documentos_filtrados:
        print("No se encontraron documentos con ese filtro.\n")
        return []

    print(f"\n{len(documentos_filtrados)} documentos encontrados:\n")
    for i, d in enumerate(documentos_filtrados, 1):
        print(f"{i}. {d['archivo']} (ID: {d['id']})")

    return documentos_filtrados


def filtrar_documentos_por_metadatos_api(corpus_id, meta_nombre, valor):
    """
    Versión no interactiva para Flask.
    Devuelve lista de documentos que cumplen el filtro (sin pedir input()).
    """
    import requests

    url = f"{BASE_URL}{corpus_id}/tabla"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json().get("data", {})

    metadatos = data.get("metadatos", [])
    tabla = data.get("tabla", [])

    # Buscar ID del metadato según su nombre
    meta_id = None
    for m in metadatos:
        if m[1].strip().lower() == meta_nombre.strip().lower():
            meta_id = m[0]
            break
    if meta_id is None:
        return []

    # Filtrar los documentos por valor
    documentos_filtrados = []
    for doc in tabla:
        doc_id, archivo, metas = doc
        for mid, val in metas:
            if mid == meta_id and val.strip().lower() == valor.strip().lower():
                documentos_filtrados.append({"id": doc_id, "archivo": archivo})
                break

    return documentos_filtrados

# =====================================
# NUEVA FUNCIÓN: FILTRAR POR VARIOS METADATOS (para API Flask/app.py)
# =====================================


def filtrar_documentos_por_varios_metadatos_api(corpus_id, metas, valores):
    """
    Filtra documentos que cumplan simultáneamente varios metadatos y valores.
    Ejemplo:
        metas = ["Área", "Lengua"]
        valores = ["Medicina", "Español"]
    Devuelve una lista de documentos (diccionarios con id y archivo).
    """
    import requests
    from c3 import BASE_URL, headers

    url = f"{BASE_URL}{corpus_id}/tabla"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json().get("data", {})

    tabla = data.get("tabla", [])
    metadatos = data.get("metadatos", [])

    # Crear mapa de id_metadato → nombre_metadato
    id_to_nombre = {m[0]: m[1] for m in metadatos}

    # Crear mapa inverso: nombre_metadato → id_metadato
    nombre_to_id = {m[1]: m[0] for m in metadatos}

    # Convertir nombres a IDs (ignorando los que no existan)
    filtros = []
    for meta, valor in zip(metas, valores):
        mid = nombre_to_id.get(meta)
        if mid:
            filtros.append((mid, valor))

    documentos_filtrados = []
    for doc in tabla:
        doc_id, archivo, metas_doc = doc
        # Crear dict rápido de {meta_id: valor}
        metas_dict = {m[0]: m[1] for m in metas_doc}
        # Comprobar que todos los pares meta=valor coinciden
        cumple_todos = all(metas_dict.get(
            mid) == valor for mid, valor in filtros)
        if cumple_todos:
            documentos_filtrados.append({"id": doc_id, "archivo": archivo})

    return documentos_filtrados


# --------------------------------------------
# NUEVAS FUNCIONES PARA DICCIONARIOS
# --------------------------------------------


def guardar_diccionario(nombre_diccionario, grafo, builder):
    """
    Guarda el grafo como diccionario nombrado en dos formatos:
    - JSON: conserva toda la información (word_contexts, vocab_freq)
    - GraphML: compatible con Gephi y herramientas externas
    """
    base_name = nombre_diccionario.replace(" ", "_")
    archivo_json = f"{base_name}.json"
    archivo_graphml = f"{base_name}.graphml"

    ruta_json = os.path.join(GRAPH_DIR, archivo_json)
    ruta_graphml = os.path.join(GRAPH_DIR, archivo_graphml)

    # ----- Guardar en formato JSON -----
    data = {
        "nombre": nombre_diccionario,
        "nodes": list(grafo.nodes(data=True)),
        "edges": list(grafo.edges(data=True)),
        "word_contexts": {k: list(v) for k, v in builder.word_contexts.items()},
        "vocab_freq": dict(builder.vocab_freq)
    }

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # ----- Guardar en formato GraphML -----
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

    if not any(d["nombre"] == nombre_diccionario for d in index):
        index.append({
            "nombre": nombre_diccionario,
            "archivo_json": archivo_json,
            "archivo_graphml": archivo_graphml
        })
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Diccionario '{nombre_diccionario}' guardado exitosamente en:")
    print(f"   • JSON: {ruta_json}")
    print(f"   • GraphML: {ruta_graphml}")


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
# PROCESAMIENTO PRINCIPAL
# --------------------------------------------
# --------------------------------------------
# PROCESAMIENTO PRINCIPAL (modo terminal)
# --------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("        DICCIONARIO INVERSO — MODO TERMINAL")
    print("=" * 60)

    # Mostrar corpus disponibles
    corpus_list = listar_corpus()
    corpus = elegir_corpus(corpus_list)
    corpus_id = corpus.get("id")

    # Mostrar documentos
    documentos = filtrar_documentos_por_metadatos(corpus_id)
    docs_sel = elegir_documentos(documentos)

    print("\nDescargando y procesando documentos seleccionados...\n")

    textos = []
    processor = TextProcessor()
    builder = GraphBuilder(processor)

    for d in docs_sel:
        try:
            txt = descargar_documento(corpus_id, d.get("id"))
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
