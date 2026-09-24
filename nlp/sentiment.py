"""
Análisis de sentimiento operacional — motor de reglas/léxico técnico.

Enfoque HÍBRIDO (ver contexto del proyecto): este motor de reglas es
explicable y corre en tiempo real; asigna una polaridad OPERACIONAL
(Negativo/Crítico, Neutro, Positivo/Favorable) que es independiente de la
severidad formal del incidente — ese es justamente el caso de negocio
central: un evento que "formalmente no paró el servicio" puede darle
"mala espina" al operador, y esa señal vale la pena capturarla aparte.

Como el dataset no trae etiqueta de sentimiento, este mismo motor generó
las etiquetas de entrenamiento para el clasificador supervisado
(TF-IDF + SVM) en train_models.py. La app muestra ambas predicciones
lado a lado.
"""
from .preprocessing import normalizar

NEGATIVAS_FUERTES = [
    "mala espina", "no me da buena espina", "no me convence", "a ciegas",
    "furios", "de milagro", "kernel panic", "colapso", "incomunicad",
    "nadie responde", "olor a quemado", "emergencia", "sin redundancia",
    "crashloopbackoff", "perdio el quorum", "perdio quorum",
    "caida total", "corte total", "bloqueadas",
]

NEGATIVAS_MODERADAS = [
    "ojo que", "ojo con", "preocup", "ya van", "reincid", "no responde",
    "vibracion anomala", "ruido anomalo", "ruido forzado", "da mala espina",
]

POSITIVAS_FUERTES = [
    "sin novedad", "sin errores", "sin incidentes", "finalizada al 100",
    "finalizado al 100", "restablecido", "se normalizo", "normalizo solo",
    "operando con normalidad",
]

POSITIVAS_MODERADAS = [
    "no interrumpe", "no afecta", "no tira trafico", "resuelto",
    "corregido", "sin problemas",
]

UMBRAL_NEGATIVO = -2
UMBRAL_POSITIVO = 2


def _buscar_senales(texto_norm: str, frases: list[str], peso: int) -> list[dict]:
    senales = []
    for frase in frases:
        if frase in texto_norm:
            senales.append({"frase": frase, "peso": peso})
    return senales


def analizar_reglas(texto: str) -> dict:
    """Devuelve {'etiqueta': ..., 'score': int, 'senales': [...]}"""
    texto_norm = normalizar(texto or "")

    senales = []
    senales += _buscar_senales(texto_norm, NEGATIVAS_FUERTES, -2)
    senales += _buscar_senales(texto_norm, NEGATIVAS_MODERADAS, -1)
    senales += _buscar_senales(texto_norm, POSITIVAS_FUERTES, 2)
    senales += _buscar_senales(texto_norm, POSITIVAS_MODERADAS, 1)

    score = sum(s["peso"] for s in senales)

    if score <= UMBRAL_NEGATIVO:
        etiqueta = "Negativo/Crítico"
    elif score >= UMBRAL_POSITIVO:
        etiqueta = "Positivo/Favorable"
    else:
        etiqueta = "Neutro"

    return {"etiqueta": etiqueta, "score": score, "senales": senales}


def etiqueta_reglas(texto: str) -> str:
    return analizar_reglas(texto)["etiqueta"]
