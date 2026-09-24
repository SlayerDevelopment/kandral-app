"""
Enriquecimiento del dataset: agrega columnas derivadas necesarias para
entrenar y para la UI (texto preprocesado, sentimiento por reglas, número
de activos mencionados).

Importante: este módulo NO se ejecuta suelto. Se importa y se llama
únicamente desde train_models.py, que es el único punto de entrada del
pipeline completo. Correrlo aparte puede dejar el CSV inconsistente con
los modelos entrenados.
"""
import pandas as pd

from nlp.preprocessing import preprocesar
from nlp.sentiment import analizar_reglas
from nlp.ner import extraer_entidades


def enriquecer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["texto_limpio"] = df["texto_bitacora"].apply(preprocesar)

    resultados_sentimiento = df["texto_bitacora"].apply(analizar_reglas)
    df["sentimiento_regla"] = resultados_sentimiento.apply(lambda r: r["etiqueta"])
    df["sentimiento_score"] = resultados_sentimiento.apply(lambda r: r["score"])

    df["num_activos_mencionados"] = df["texto_bitacora"].apply(
        lambda t: sum(1 for e in extraer_entidades(t) if e["tipo"] == "ACTIVO_EQUIPO")
    )

    return df
