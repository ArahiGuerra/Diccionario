

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
from c3 import cargar_diccionario, ReverseDict, GRAPH_DIR

app = Flask(__name__)
# Permitir peticiones desde cualquier dominio (ajusta según necesites)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Cache de diccionarios cargados para evitar recargarlos constantemente
diccionarios_cache = {}


def get_diccionario(nombre):
    """Obtiene un diccionario del cache o lo carga si no existe."""
    if nombre not in diccionarios_cache:
        grafo, processor, builder = cargar_diccionario(nombre)
        if grafo is None:
            return None
        diccionarios_cache[nombre] = {
            "grafo": grafo,
            "processor": processor,
            "builder": builder,
            "reverse_dict": ReverseDict(grafo, processor, builder)
        }
    return diccionarios_cache[nombre]


# ============================================
# ENDPOINTS PÚBLICOS
# ============================================

@app.route("/api/v1/diccionarios", methods=["GET"])
def listar_diccionarios():
    """
    Lista todos los diccionarios disponibles.
    
    Respuesta:
    {
        "ok": true,
        "diccionarios": [
            {
                "nombre": "corpus_medicina",
                "nodos": 5000,
                "aristas": 25000
            }
        ]
    }
    """
    try:
        index_path = os.path.join(GRAPH_DIR, "diccionarios_index.json")
        if not os.path.exists(index_path):
            return jsonify({"ok": True, "diccionarios": []})
        
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        
        # Agregar información adicional de cada diccionario
        resultado = []
        for dic in index:
            nombre = dic["nombre"]
            dic_obj = get_diccionario(nombre)
            if dic_obj:
                resultado.append({
                    "nombre": nombre,
                    "nodos": len(dic_obj["grafo"].nodes()),
                    "aristas": len(dic_obj["grafo"].edges())
                })
        
        return jsonify({"ok": True, "diccionarios": resultado})
    
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/v1/buscar", methods=["POST"])
def buscar():
    """
    Busca palabras basándose en una definición.
    
    Body (JSON):
    {
        "diccionario": "corpus_medicina",
        "definicion": "órgano que bombea sangre",
        "top_k": 10
    }
    
    Respuesta:
    {
        "ok": true,
        "diccionario": "corpus_medicina",
        "definicion": "órgano que bombea sangre",
        "resultados": [
            {"palabra": "corazón", "score": 0.8543},
            {"palabra": "cardíaco", "score": 0.7231}
        ]
    }
    """
    try:
        data = request.get_json()
        
        # Validar parámetros
        diccionario_nombre = data.get("diccionario")
        definicion = data.get("definicion", "").strip()
        top_k = int(data.get("top_k", 10))
        
        if not diccionario_nombre:
            return jsonify({
                "ok": False, 
                "error": "Falta el parámetro 'diccionario'"
            }), 400
        
        if not definicion:
            return jsonify({
                "ok": False, 
                "error": "Falta el parámetro 'definicion'"
            }), 400
        
        if top_k < 1 or top_k > 50:
            return jsonify({
                "ok": False, 
                "error": "top_k debe estar entre 1 y 50"
            }), 400
        
        # Cargar diccionario
        dic = get_diccionario(diccionario_nombre)
        if dic is None:
            return jsonify({
                "ok": False, 
                "error": f"Diccionario '{diccionario_nombre}' no encontrado"
            }), 404
        
        # Realizar búsqueda
        resultados = dic["reverse_dict"].buscar_multiple_estrategias(
            definicion, 
            top_k=top_k
        )
        
        # Formatear resultados
        resultados_formateados = [
            {"palabra": r[0], "score": round(float(r[1]), 4)} 
            for r in resultados
        ]
        
        return jsonify({
            "ok": True,
            "diccionario": diccionario_nombre,
            "definicion": definicion,
            "resultados": resultados_formateados
        })
    
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/v1/buscar_batch", methods=["POST"])
def buscar_batch():
    """
    Busca múltiples definiciones en un solo request.
    
    Body (JSON):
    {
        "diccionario": "corpus_medicina",
        "definiciones": [
            "órgano que bombea sangre",
            "líquido rojo vital"
        ],
        "top_k": 5
    }
    
    Respuesta:
    {
        "ok": true,
        "resultados": [
            {
                "definicion": "órgano que bombea sangre",
                "palabras": [...]
            }
        ]
    }
    """
    try:
        data = request.get_json()
        
        diccionario_nombre = data.get("diccionario")
        definiciones = data.get("definiciones", [])
        top_k = int(data.get("top_k", 10))
        
        if not diccionario_nombre or not definiciones:
            return jsonify({
                "ok": False, 
                "error": "Faltan parámetros requeridos"
            }), 400
        
        if len(definiciones) > 20:
            return jsonify({
                "ok": False, 
                "error": "Máximo 20 definiciones por request"
            }), 400
        
        # Cargar diccionario
        dic = get_diccionario(diccionario_nombre)
        if dic is None:
            return jsonify({
                "ok": False, 
                "error": f"Diccionario '{diccionario_nombre}' no encontrado"
            }), 404
        
        # Procesar todas las definiciones
        resultados_batch = []
        for definicion in definiciones:
            resultados = dic["reverse_dict"].buscar_multiple_estrategias(
                definicion, 
                top_k=top_k
            )
            resultados_batch.append({
                "definicion": definicion,
                "palabras": [
                    {"palabra": r[0], "score": round(float(r[1]), 4)} 
                    for r in resultados
                ]
            })
        
        return jsonify({
            "ok": True,
            "diccionario": diccionario_nombre,
            "resultados": resultados_batch
        })
    
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/v1/info/<nombre_diccionario>", methods=["GET"])
def info_diccionario(nombre_diccionario):
    """
    Obtiene información detallada de un diccionario.
    
    Respuesta:
    {
        "ok": true,
        "nombre": "corpus_medicina",
        "nodos": 5000,
        "aristas": 25000,
        "palabras_frecuentes": [
            {"palabra": "salud", "frecuencia": 150},
            {"palabra": "paciente", "frecuencia": 120}
        ]
    }
    """
    try:
        dic = get_diccionario(nombre_diccionario)
        if dic is None:
            return jsonify({
                "ok": False, 
                "error": f"Diccionario '{nombre_diccionario}' no encontrado"
            }), 404
        
        # Obtener palabras más frecuentes
        palabras_frecuentes = [
            {"palabra": palabra, "frecuencia": int(freq)}
            for palabra, freq in dic["builder"].vocab_freq.most_common(20)
        ]
        
        return jsonify({
            "ok": True,
            "nombre": nombre_diccionario,
            "nodos": len(dic["grafo"].nodes()),
            "aristas": len(dic["grafo"].edges()),
            "palabras_frecuentes": palabras_frecuentes
        })
    
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/v1/health", methods=["GET"])
def health():
    """Endpoint para verificar que la API está funcionando."""
    return jsonify({
        "ok": True,
        "status": "running",
        "version": "1.0.0"
    })


@app.route("/api/v1/docs", methods=["GET"])
def docs():
    """Documentación de la API."""
    return jsonify({
        "version": "1.0.0",
        "endpoints": {
            "GET /api/v1/diccionarios": "Lista todos los diccionarios disponibles",
            "POST /api/v1/buscar": "Busca palabras basándose en una definición",
            "POST /api/v1/buscar_batch": "Busca múltiples definiciones",
            "GET /api/v1/info/<nombre>": "Información detallada de un diccionario",
            "GET /api/v1/health": "Verifica que la API esté funcionando"
        },
        "ejemplos": {
            "buscar": {
                "url": "/api/v1/buscar",
                "method": "POST",
                "body": {
                    "diccionario": "corpus_medicina",
                    "definicion": "órgano que bombea sangre",
                    "top_k": 10
                }
            }
        }
    })


# ============================================
# CONFIGURACIÓN Y EJECUCIÓN
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("    API PÚBLICA DE BÚSQUEDAS - DICCIONARIO INVERSO")
    print("=" * 60)
    print("\nEndpoints disponibles:")
    print("  • GET  /api/v1/diccionarios")
    print("  • POST /api/v1/buscar")
    print("  • POST /api/v1/buscar_batch")
    print("  • GET  /api/v1/info/<nombre>")
    print("  • GET  /api/v1/health")
    print("  • GET  /api/v1/docs")
    print("\n" + "=" * 60)
    
    # Precargar diccionarios al inicio (opcional)
    print("\nPrecargando diccionarios...")
    index_path = os.path.join(GRAPH_DIR, "diccionarios_index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        for dic in index[:3]:  # Precargar solo los primeros 3
            nombre = dic["nombre"]
            print(f"  • Cargando '{nombre}'...")
            get_diccionario(nombre)
    
    print("\n✓ API lista en http://localhost:5001")
    print("=" * 60)
    
    app.run(debug=False, host="0.0.0.0", port=5001)