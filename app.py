from flask import Flask, render_template, jsonify, request
import os
import json
import networkx as nx
from collections import Counter
from c3 import (
    listar_corpus,
    listar_documentos,
    descargar_documento,
    TextProcessor,
    GraphBuilder,
    ReverseDict,
    TEXTS_DIR,
    LEMAS_DIR,
    GRAPH_DIR,
    cargar_diccionario,
)

url_prefix = ''
with open('config.json', 'r') as f:
    config = json.load(f)
    url_prefix = config.get('url_prefix', '')

if url_prefix:
    static_url_path = '/' + url_prefix + '/static'
else:
    static_url_path = '/static'

app = Flask(__name__, static_folder="static", static_url_path=static_url_path, template_folder="templates")
if url_prefix:
    app.config['APPLICATION_ROOT'] = url_prefix + '/'

state = {
    "status": "idle",
    "message": "",
    "current_graph": None,
    "builder": None,
    "processor": None,
    "reverse_dict": None,
    "last_graph_file": None,
    "current_diccionario": None,
}


def graph_to_json(G, top_n_nodes=None):
    nodes = []
    edges = []
    nodes_list = list(G.nodes())
    if top_n_nodes:
        try:
            nodes_sorted = sorted(
                G.nodes(data=True),
                key=lambda x: x[1].get("frequency", 0),
                reverse=True,
            )[:top_n_nodes]
            nodes_list = [n for n, _ in nodes_sorted]
        except Exception:
            pass

    for n in nodes_list:
        data = G.nodes[n]
        nodes.append(
            {
                "id": n,
                "frequency": int(data.get("frequency", 0)),
                "degree": int(data.get("degree", G.degree(n))),
            }
        )

    for u, v, d in G.edges(data=True):
        if u in nodes_list and v in nodes_list:
            edges.append(
                {"source": u, "target": v,
                    "weight": float(d.get("weight", 1.0))}
            )

    return {"nodes": nodes, "edges": edges}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/corpora", methods=["GET"])
def api_corpora():
    try:
        corpus_list = listar_corpus()
        simplified = [
            {"id": c["id"], "nombre": c.get("nombre", c.get("titulo", ""))}
            for c in corpus_list
        ]
        return jsonify({"ok": True, "data": simplified})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# API: listar documentos de un corpus (ahora con filtrado opcional por metadatos)
@app.route("/api/documentos/<int:corpus_id>", methods=["GET"])
def api_documentos(corpus_id):
    """
    Obtiene documentos de un corpus, con opción de filtrar por uno o varios metadatos.
    - Sin filtros: /api/documentos/131
    - Filtro simple: /api/documentos/131?meta=Área&valor=Medicina
    - Múltiples filtros: /api/documentos/131?meta=Área,Lengua&valor=Medicina,Español
    """
    try:
        meta_param = request.args.get("meta")
        valor_param = request.args.get("valor")

        # Caso: múltiples filtros separados por coma
        if meta_param and valor_param:
            metas = [m.strip() for m in meta_param.split(",")]
            valores = [v.strip() for v in valor_param.split(",")]

            # Si hay más de uno, usamos la nueva función
            if len(metas) > 1:
                from c3 import filtrar_documentos_por_varios_metadatos_api
                documentos = filtrar_documentos_por_varios_metadatos_api(
                    corpus_id, metas, valores
                )
                simplified = [
                    {"id": d["id"], "archivo": d["archivo"]} for d in documentos
                ]
                return jsonify({"ok": True, "data": simplified, "filtered": True})

            # Si hay solo uno, usamos la función clásica
            from c3 import filtrar_documentos_por_metadatos_api
            documentos = filtrar_documentos_por_metadatos_api(
                corpus_id, metas[0], valores[0]
            )
            simplified = [{"id": d["id"], "archivo": d["archivo"]}
                          for d in documentos]
            return jsonify({"ok": True, "data": simplified, "filtered": True})

        # Si no se enviaron parámetros, listamos todos los documentos
        from c3 import listar_documentos
        documentos = listar_documentos(corpus_id)
        simplified = [
            {"id": d["id"], "archivo": d.get("archivo", str(d.get("id")))}
            for d in documentos
        ]
        return jsonify({"ok": True, "data": simplified, "filtered": False})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# API: obtener metadatos y sus valores posibles de un corpus
@app.route("/api/metadatos/<int:corpus_id>", methods=["GET"])
def api_metadatos(corpus_id):
    """
    Devuelve los metadatos y sus valores posibles.
    Ejemplo de respuesta:
    {
      "ok": true,
      "data": [
        {"nombre": "Área", "valores": ["Medicina", "Ingeniería", "COVID"]}
      ]
    }
    """
    try:
        from c3 import client
        docs = client.docs_tabla(corpus_id)

        metadatos = {}
        for d in docs:
            for metadato in d["metadata"]:
                metadatos.setdefault(metadato, set()).add(d["metadata"][metadato])

        resultado = [{"nombre": meta, "valores": sorted(valores)} for meta, valores in metadatos.items()]

        return jsonify({"ok": True, "data": resultado})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/process", methods=["POST"])
def api_process():
    data = request.get_json()
    corpus_id = data.get("corpus_id")
    doc_ids = data.get("doc_ids", [])
    dic_name = data.get("dic_name", "Diccionario_Sin_Nombre")

    if not corpus_id or not doc_ids:
        return jsonify({"ok": False, "error": "Faltan corpus_id o doc_ids"}), 400

    try:
        state["status"] = "processing"
        state["message"] = "Procesando documentos..."

        processor = TextProcessor()
        builder = GraphBuilder(processor)
        textos = []

        for doc_id in doc_ids:
            txt = descargar_documento(corpus_id, doc_id)
            txt_clean = processor.limpiar_texto_avanzado(txt)
            textos.append(txt_clean)

        texto_unido = " ".join(textos)
        tokens_procesados = processor.lematizar_con_spacy(texto_unido)
        grafo = builder.construir_grafo_mejorado(tokens_procesados)

        from c3 import guardar_diccionario
        guardar_diccionario(dic_name, grafo, builder)

        reverse_dict = ReverseDict(grafo, processor, builder)

        state.update(
            {
                "status": "done",
                "message": f"Diccionario '{dic_name}' generado exitosamente.",
                "current_graph": grafo,
                "builder": builder,
                "processor": processor,
                "reverse_dict": reverse_dict,
                "current_diccionario": dic_name,
            }
        )

        graph_json = graph_to_json(grafo, top_n_nodes=500)
        return jsonify({"ok": True, "graph": graph_json, "message": state["message"]})
    except Exception as e:
        state["status"] = "error"
        state["message"] = str(e)
        return jsonify({"ok": False, "error": str(e)}), 500


# Listar diccionarios disponibles
@app.route("/api/diccionarios", methods=["GET"])
def api_diccionarios():
    index_path = os.path.join(GRAPH_DIR, "diccionarios_index.json")
    if not os.path.exists(index_path):
        return jsonify({"ok": True, "data": []})
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify({"ok": True, "data": data})


# Seleccionar y cargar un diccionario existente
@app.route("/api/load_diccionario", methods=["POST"])
def api_load_diccionario():
    data = request.get_json()
    nombre = data.get("nombre")
    if not nombre:
        return jsonify({"ok": False, "error": "Falta el nombre del diccionario."}), 400

    grafo, processor, builder = cargar_diccionario(nombre)
    if grafo is None:
        return jsonify({"ok": False, "error": "No se pudo cargar el diccionario."}), 404

    reverse_dict = ReverseDict(grafo, processor, builder)
    state.update(
        {
            "current_graph": grafo,
            "builder": builder,
            "processor": processor,
            "reverse_dict": reverse_dict,
            "current_diccionario": nombre,
        }
    )

    return jsonify(
        {
            "ok": True,
            "message": f"Diccionario '{nombre}' cargado correctamente.",
            "graph": graph_to_json(grafo, top_n_nodes=500),
        }
    )


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json()
    definition = data.get("definition", "")
    dic_name = data.get("diccionario")
    top_k = int(data.get("top_k", 10))

    if not definition:
        return jsonify({"ok": False, "error": "Falta la definición."}), 400

    if dic_name:
        grafo, processor, builder = cargar_diccionario(dic_name)
        if not grafo:
            return jsonify({"ok": False, "error": "Diccionario no encontrado."}), 404
        rd = ReverseDict(grafo, processor, builder)
    else:
        rd = state.get("reverse_dict")
        if rd is None:
            return jsonify({"ok": False, "error": "No hay diccionario cargado."}), 400

    resultados = rd.buscar_multiple_estrategias(definition, top_k=top_k)
    resultados_s = [
        {"palabra": r[0], "score": float(r[1])} for r in resultados]
    return jsonify({"ok": True, "results": resultados_s})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
