"""
NER por reglas + diccionario (gazetteer) para bitácoras de Kandral.

No usa deep learning: los nombres de activos en Kandral tienen formato
muy estructurado (SW-CORE-01, INC-46412), así que reglas + diccionario
son precisas, explicables y no requieren entrenamiento ni GPU.

Tipologías reconocidas:
- ACTIVO_EQUIPO: códigos de equipo (switches, servidores, UPS, etc.)
- UBICACION_FISICA: racks, salas, pasillos, segmentos de red
- TICKET_GESTION: tickets INC-/CHG-/REQ-
- PARAMETRO_METRICA: valores numéricos con unidad técnica
"""
import re

# Prefijos de activo observados en el dataset de Kandral (diccionario).
# Se usan para marcar confianza "alta"; cualquier otro código con el mismo
# formato (2-10 letras + 1-3 segmentos alfanuméricos) igual se reconoce,
# pero con confianza "media", para generalizar a equipos no vistos.
PREFIJOS_ACTIVO_CONOCIDOS = {
    "SW", "SRV", "ESXI", "SAN", "CHILLER", "RTR", "UPS", "LUN", "TAPE",
    "NAS", "GEN", "FW", "RAID", "EXTINCTION", "HVAC", "PDU", "DB",
}

RE_TICKET = re.compile(r"\b(?:INC|CHG|REQ|TCK)-\d{3,6}\b")
RE_ACTIVO = re.compile(r"\b[A-Z]{2,10}(?:-[A-Z0-9]+){1,3}\b")

RE_UBICACION = re.compile(
    r"\b(?:"
    r"rack\s+[A-Z]?-?\d+"
    r"|sala\s+(?:de\s+)?[a-záéíóúñ]+(?:\s+[a-záéíóúñ]+)?"
    r"|pasillo\s+(?:fr[ií]o|caliente)\s*\d*"
    r"|piso\s+\d+"
    r"|cuarto\s+de\s+[a-záéíóúñ]+"
    r"|anillo\s+metropolitano"
    r"|DMZ"
    r"|VLAN\s?\d+"
    r")\b",
    flags=re.IGNORECASE,
)

RE_METRICA_NUM = re.compile(
    r"\b\d+[.,]?\d*\s?(?:°c|%|ms|mbps|gbps|gb|tb|mb|kb|db|rpm|kw|w|v)\b",
    flags=re.IGNORECASE,
)
KEYWORDS_METRICA = [
    "latencia", "temperatura", "voltaje", "ancho de banda", "uso de cpu",
    "uso de memoria", "disk queue", "quorum", "descarte de paquetes",
    "iops", "throughput",
]
RE_METRICA_KEYWORD = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in KEYWORDS_METRICA) + r")\b",
    flags=re.IGNORECASE,
)

# Orden de prioridad al resolver solapamientos entre tipologías
PRIORIDAD = {
    "TICKET_GESTION": 4,
    "ACTIVO_EQUIPO": 3,
    "UBICACION_FISICA": 2,
    "PARAMETRO_METRICA": 1,
}


def _candidatos(texto: str) -> list[dict]:
    candidatos = []

    for m in RE_TICKET.finditer(texto):
        candidatos.append(dict(tipo="TICKET_GESTION", texto=m.group(0),
                                inicio=m.start(), fin=m.end(), confianza="alta"))

    for m in RE_ACTIVO.finditer(texto):
        prefijo = m.group(0).split("-")[0]
        confianza = "alta" if prefijo in PREFIJOS_ACTIVO_CONOCIDOS else "media"
        candidatos.append(dict(tipo="ACTIVO_EQUIPO", texto=m.group(0),
                                inicio=m.start(), fin=m.end(), confianza=confianza))

    for m in RE_UBICACION.finditer(texto):
        candidatos.append(dict(tipo="UBICACION_FISICA", texto=m.group(0),
                                inicio=m.start(), fin=m.end(), confianza="alta"))

    for m in RE_METRICA_NUM.finditer(texto):
        candidatos.append(dict(tipo="PARAMETRO_METRICA", texto=m.group(0),
                                inicio=m.start(), fin=m.end(), confianza="alta"))
    for m in RE_METRICA_KEYWORD.finditer(texto):
        candidatos.append(dict(tipo="PARAMETRO_METRICA", texto=m.group(0),
                                inicio=m.start(), fin=m.end(), confianza="media"))

    return candidatos


def extraer_entidades(texto: str) -> list[dict]:
    """Devuelve entidades no solapadas, ordenadas por posición en el texto.
    Ante solapamiento, gana la tipología de mayor prioridad (ticket >
    activo > ubicación > métrica) y, en empate, la coincidencia más larga.
    """
    texto = texto or ""
    candidatos = _candidatos(texto)
    candidatos.sort(key=lambda c: (c["inicio"], -PRIORIDAD[c["tipo"]], -(c["fin"] - c["inicio"])))

    seleccionadas = []
    ocupado = []  # lista de (inicio, fin) ya tomados

    for c in candidatos:
        if any(c["inicio"] < fin and c["fin"] > ini for ini, fin in ocupado):
            continue
        seleccionadas.append(c)
        ocupado.append((c["inicio"], c["fin"]))

    seleccionadas.sort(key=lambda c: c["inicio"])
    return seleccionadas


def resumir_por_tipo(entidades: list[dict]) -> dict[str, list[str]]:
    """Agrupa los textos detectados por tipología, sin duplicados,
    preservando el orden de aparición."""
    agrupado: dict[str, list[str]] = {}
    for e in entidades:
        lista = agrupado.setdefault(e["tipo"], [])
        if e["texto"] not in lista:
            lista.append(e["texto"])
    return agrupado
