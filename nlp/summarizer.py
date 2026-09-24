"""
Resumen automático estructurado de turno.

Extractivo, basado en reglas sobre los resultados de los otros 4 modelos
(no LLM generativo, para que el prototipo sea autocontenido). Genera 3
secciones: Estado Operacional Global (semáforo Rojo/Amarillo/Verde),
Incidentes Cerrados, Tareas Pendientes Priorizadas por severidad.

Nota: el dataset no trae columna de fecha/guardia puntual (decisión del
usuario), así que el resumen agrega TODOS los registros históricos de un
turno (Mañana/Tarde/Noche), no una guardia específica.
"""
from .preprocessing import normalizar

ORDEN_SEVERIDAD = {"P1 - Crítica": 0, "P2 - Alta": 1, "P3 - Media": 2, "P4 - Baja": 3}

PALABRAS_RESOLUCION = [
    "restablecido", "resuelto", "corregido", "finalizada al 100",
    "finalizado al 100", "sin novedad", "se normalizo", "normalizo solo",
    "sin errores", "sin incidentes",
]


def _esta_cerrado(row) -> bool:
    if row.get("sentimiento_regla") == "Positivo/Favorable":
        return True
    texto_norm = normalizar(row.get("texto_bitacora", "") or "")
    return any(p in texto_norm for p in PALABRAS_RESOLUCION)


def _semaforo(pendientes: list[dict], total: int) -> str:
    if total == 0:
        return "Verde"
    n_p1 = sum(1 for r in pendientes if r["severidad"] == "P1 - Crítica")
    n_p2 = sum(1 for r in pendientes if r["severidad"] == "P2 - Alta")
    negativos = sum(1 for r in pendientes if r["sentimiento_regla"] == "Negativo/Crítico")
    ratio_negativo = negativos / total

    if n_p1 >= 1 or ratio_negativo > 0.5:
        return "Rojo"
    if n_p2 >= 1 or ratio_negativo > 0.25:
        return "Amarillo"
    return "Verde"


def generar_resumen(turno: str, registros: list[dict]) -> dict:
    """`registros` es una lista de dicts con al menos: id, area_tecnica,
    severidad, activo_principal, ticket_asociado, texto_bitacora,
    sentimiento_regla (ya enriquecido)."""
    total = len(registros)

    cerrados, pendientes = [], []
    for r in registros:
        (cerrados if _esta_cerrado(r) else pendientes).append(r)

    pendientes.sort(key=lambda r: ORDEN_SEVERIDAD.get(r["severidad"], 9))

    return {
        "turno": turno,
        "total_registros": total,
        "estado_operacional": {
            "semaforo": _semaforo(pendientes, total),
            "total_pendientes": len(pendientes),
            "total_cerrados": len(cerrados),
        },
        "incidentes_cerrados": [
            {
                "id": r["id"], "area_tecnica": r["area_tecnica"],
                "severidad": r["severidad"], "activo_principal": r["activo_principal"],
                "ticket_asociado": r["ticket_asociado"],
            }
            for r in cerrados
        ],
        "tareas_pendientes": [
            {
                "id": r["id"], "area_tecnica": r["area_tecnica"],
                "severidad": r["severidad"], "activo_principal": r["activo_principal"],
                "ticket_asociado": r["ticket_asociado"],
                "sentimiento_regla": r["sentimiento_regla"],
            }
            for r in pendientes
        ],
        "validacion_humana_obligatoria": (
            "Este resumen es generado automáticamente a partir de reglas sobre "
            "los modelos de NLP del prototipo. No reemplaza el criterio del "
            "supervisor de turno: toda acción debe ser validada por una persona "
            "antes de cerrarse (principio de Trabajo Aumentado / Augmented Working)."
        ),
    }
