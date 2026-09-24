"""
Backend Flask del prototipo Kandral PLN.

Endpoints:
- POST /api/analizar          analiza un texto suelto con los 4 modelos
- GET  /api/dataset            lista paginada/filtrable del dataset
- GET  /api/dataset/opciones   valores disponibles para filtros
- POST /api/resumen-turno      genera el resumen estructurado de un turno
- GET  /api/metrics            métricas de evaluación de los clasificadores
"""
import json
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from nlp.preprocessing import preprocesar
from nlp.ner import extraer_entidades, resumir_por_tipo
from nlp.sentiment import analizar_reglas
from nlp.summarizer import generar_resumen
from chatbot import ChatbotKandral

load_dotenv()

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATASET_CSV = BASE_DIR / "data" / "dataset_enriquecido.csv"

ORDEN_SEVERIDAD = ["P1 - Crítica", "P2 - Alta", "P3 - Media", "P4 - Baja"]
ORDEN_TURNO = ["Mañana", "Tarde", "Noche"]

app = Flask(__name__)

_modelos = {}
_df = None
_metrics = None
_chatbot = None


def cargar_estado():
    global _df, _metrics, _chatbot
    for nombre, archivo in [
        ("area", "area_clf.pkl"),
        ("severidad", "severidad_clf.pkl"),
        ("sentimiento", "sentimiento_clf.pkl"),
    ]:
        ruta = MODELS_DIR / archivo
        if not ruta.exists():
            raise RuntimeError(
                f"No se encontró {archivo}. Corré 'python train_models.py' primero."
            )
        _modelos[nombre] = joblib.load(ruta)

    if not DATASET_CSV.exists():
        raise RuntimeError(
            "No se encontró el dataset enriquecido. Corré 'python train_models.py' primero."
        )
    _df = pd.read_csv(DATASET_CSV)
    _df["ticket_asociado"] = _df["ticket_asociado"].where(_df["ticket_asociado"].notna(), None)

    with open(MODELS_DIR / "metrics.json", "r", encoding="utf-8") as f:
        _metrics = json.load(f)

    _chatbot = ChatbotKandral(df=_df, modelos=_modelos)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analizar", methods=["POST"])
def api_analizar():
    body = request.get_json(silent=True) or {}
    texto = (body.get("texto") or "").strip()
    if not texto:
        return jsonify({"error": "El campo 'texto' es obligatorio."}), 400

    texto_limpio = preprocesar(texto)
    entidades = extraer_entidades(texto)
    reglas = analizar_reglas(texto)

    respuesta = {
        "texto_original": texto,
        "entidades": entidades,
        "entidades_por_tipo": resumir_por_tipo(entidades),
        "area_tecnica": _modelos["area"].predict([texto_limpio])[0],
        "severidad": _modelos["severidad"].predict([texto_limpio])[0],
        "sentimiento": {
            "reglas": {
                "etiqueta": reglas["etiqueta"],
                "score": reglas["score"],
                "senales": reglas["senales"],
            },
            "clasificador": _modelos["sentimiento"].predict([texto_limpio])[0],
        },
    }
    return jsonify(respuesta)


@app.route("/api/dataset", methods=["GET"])
def api_dataset():
    df = _df

    turno = request.args.get("turno")
    area_tecnica = request.args.get("area_tecnica")
    severidad = request.args.get("severidad")
    sentimiento = request.args.get("sentimiento_regla")
    busqueda = request.args.get("q")

    if turno:
        df = df[df["turno"] == turno]
    if area_tecnica:
        df = df[df["area_tecnica"] == area_tecnica]
    if severidad:
        df = df[df["severidad"] == severidad]
    if sentimiento:
        df = df[df["sentimiento_regla"] == sentimiento]
    if busqueda:
        df = df[df["texto_bitacora"].str.contains(busqueda, case=False, na=False)]

    try:
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 20))))
    except ValueError:
        return jsonify({"error": "page y page_size deben ser enteros."}), 400

    total = len(df)
    total_paginas = max(1, math.ceil(total / page_size))
    inicio = (page - 1) * page_size

    columnas = [
        "id", "turno", "area_tecnica", "severidad", "activo_principal",
        "ticket_asociado", "texto_bitacora", "sentimiento_regla",
        "num_activos_mencionados",
    ]
    pagina = df[columnas].iloc[inicio: inicio + page_size]
    pagina = pagina.astype(object).where(pd.notnull(pagina), None)

    return jsonify({
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_paginas": total_paginas,
        "resultados": pagina.to_dict(orient="records"),
    })


@app.route("/api/dataset/opciones", methods=["GET"])
def api_dataset_opciones():
    df = _df
    severidades = [s for s in ORDEN_SEVERIDAD if s in df["severidad"].unique()]
    turnos = [t for t in ORDEN_TURNO if t in df["turno"].unique()]
    return jsonify({
        "turno": turnos,
        "area_tecnica": sorted(df["area_tecnica"].dropna().unique().tolist()),
        "severidad": severidades,
        "sentimiento_regla": sorted(df["sentimiento_regla"].dropna().unique().tolist()),
    })


@app.route("/api/resumen-turno", methods=["POST"])
def api_resumen_turno():
    body = request.get_json(silent=True) or {}
    turno = body.get("turno")
    if turno not in ORDEN_TURNO:
        return jsonify({"error": f"'turno' debe ser uno de {ORDEN_TURNO}."}), 400

    columnas = [
        "id", "area_tecnica", "severidad", "activo_principal",
        "ticket_asociado", "texto_bitacora", "sentimiento_regla",
    ]
    subset = _df[_df["turno"] == turno][columnas]
    subset = subset.astype(object).where(pd.notnull(subset), None)
    registros = subset.to_dict(orient="records")

    return jsonify(generar_resumen(turno, registros))


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    return jsonify(_metrics)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(silent=True) or {}
    mensaje = body.get("mensaje", "")
    historial = body.get("historial", [])

    try:
        resultado = _chatbot.responder(mensaje, historial)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Error al conectar con el modelo de lenguaje: {e}"}), 502

    return jsonify(resultado)


cargar_estado()

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=puerto, debug=True)
