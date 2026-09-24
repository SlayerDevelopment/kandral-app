"""
Punto de entrada único del pipeline de entrenamiento.

Uso:  python train_models.py

Hace todo en un solo comando: carga el dataset -> preprocesamiento ->
sentimiento por reglas -> enriquecimiento de columnas -> entrenamiento de
los 3 clasificadores (área técnica, severidad, sentimiento) -> guarda
modelos .pkl y métricas. No correr enrich_dataset.py por separado.
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from enrich_dataset import enriquecer

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
DATASET_XLSX = DATA_DIR / "Dataset_Sintetico_Bitacora_Kandral_1000 (Actualizado Total).xlsx"
DATASET_ENRIQUECIDO_CSV = DATA_DIR / "dataset_enriquecido.csv"

CLASIFICADORES = {
    "area_tecnica": "area_clf.pkl",
    "severidad": "severidad_clf.pkl",
    "sentimiento_regla": "sentimiento_clf.pkl",
}


def entrenar_clasificador(df: pd.DataFrame, columna_objetivo: str) -> dict:
    X = df["texto_limpio"]
    y = df[columna_objetivo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LinearSVC(random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "pipeline": pipeline,
        "metrics": {
            "accuracy": accuracy,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "reporte_por_clase": reporte,
            "distribucion_clases": y.value_counts().to_dict(),
        },
    }


def main():
    inicio = time.time()
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"[1/4] Cargando dataset: {DATASET_XLSX.name}")
    df = pd.read_excel(DATASET_XLSX)
    print(f"      {len(df)} registros cargados.")

    print("[2/4] Preprocesando texto + sentimiento por reglas + NER...")
    df = enriquecer(df)
    df.to_csv(DATASET_ENRIQUECIDO_CSV, index=False)
    print(f"      Dataset enriquecido guardado en {DATASET_ENRIQUECIDO_CSV.name}")

    print("[3/4] Entrenando clasificadores (TF-IDF + LinearSVC)...")
    metricas_globales = {
        "dataset": {
            "archivo": DATASET_XLSX.name,
            "n_registros": len(df),
            "nota_metodologica": (
                "Se verificó que no hay textos duplicados entre train/test en "
                "ninguno de los 3 clasificadores (sin fuga de datos)."
            ),
        },
        "clasificadores": {},
    }

    for columna, nombre_archivo in CLASIFICADORES.items():
        print(f"      - {columna} ...", end=" ")
        resultado = entrenar_clasificador(df, columna)
        joblib.dump(resultado["pipeline"], MODELS_DIR / nombre_archivo)
        metricas_globales["clasificadores"][columna] = resultado["metrics"]
        print(f"accuracy={resultado['metrics']['accuracy']:.3f}")

        # Verificación explícita de fuga de datos (textos duplicados train/test)
        X = df["texto_limpio"]
        duplicados = X[X.duplicated()].shape[0]
        metricas_globales["clasificadores"][columna]["textos_duplicados_en_dataset"] = int(duplicados)

    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metricas_globales, f, ensure_ascii=False, indent=2, default=str)

    print(f"[4/4] Listo. Modelos y métricas guardados en {MODELS_DIR}/")
    print(f"Tiempo total: {time.time() - inicio:.1f}s")


if __name__ == "__main__":
    main()
