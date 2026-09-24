"""
Preprocesamiento de texto para las bitácoras de Kandral.

Pipeline propio sin spaCy: limpieza de ruido, normalización (minúsculas,
tildes), protección de tokens compuestos (códigos de activo tipo
SW-CORE-01, métricas tipo 21°C/115ms), lematización ligera por reglas de
sufijos para verbos del dominio, y manejo de negaciones ("no" se fusiona
con el token siguiente para preservar la polaridad técnica negativa).
"""
import re
import unicodedata

TABLA_ACENTOS = str.maketrans("áéíóúÁÉÍÓÚñÑüÜ", "aeiouAEIOUnNuU")

# Códigos de activo / ticket: 2-10 letras + 1-3 segmentos alfanuméricos
# separados por guion (SW-CORE-01, INC-46412, ESXI-KANDRAL-05, SAN-02...)
RE_COMPUESTO = re.compile(r"\b[A-Za-z]{2,10}(?:-[A-Za-z0-9]+){1,3}\b")
# Métricas numéricas con unidad (21°C, 115ms, 4TB, 89%...)
RE_METRICA = re.compile(
    r"\b\d+[.,]?\d*\s?(?:°c|c°|%|ms|mbps|gbps|gb|tb|mb|kb|db|rpm|kw|w|v|a)\b",
    flags=re.IGNORECASE,
)
RE_RUIDO = re.compile(r"[^\w\sáéíóúñü°%.,-]", flags=re.IGNORECASE)
RE_ESPACIOS = re.compile(r"\s+")

# Sufijos verbales del dominio, de más largo a más corto (lematización ligera)
SUFIJOS_VERBALES = [
    "izaciones", "izacion", "amente",
    "ando", "iendo",
    "adas", "idas", "ados", "idos",
    "ada", "ida", "ado", "ido",
    "aron", "ieron", "aba", "ia",
]

# Palabras que NO deben perder su "no" fusionado aunque sean muy cortas
STOPWORDS_MINIMAS = {"de", "la", "el", "en", "y", "a", "que", "un", "una", "los", "las"}


def limpiar_texto(texto: str) -> str:
    """Quita ruido (caracteres extraños, espacios repetidos) preservando
    tildes, guiones, puntos, porcentaje y grado (se necesitan para detectar
    métricas y códigos de activo antes de normalizar)."""
    texto = texto or ""
    texto = RE_RUIDO.sub(" ", texto)
    texto = RE_ESPACIOS.sub(" ", texto).strip()
    return texto


def normalizar(texto: str) -> str:
    """Minúsculas + elimina tildes (conserva ñ vía tabla propia)."""
    return texto.translate(TABLA_ACENTOS).lower()


def _proteger_compuesto(match: re.Match) -> str:
    return match.group(0).replace("-", "_")


def _proteger_metrica(match: re.Match) -> str:
    return re.sub(r"[°\s]", "", match.group(0))


def proteger_tokens_compuestos(texto: str) -> str:
    """Convierte códigos de activo (SW-CORE-01 -> SW_CORE_01) y métricas
    (21°C -> 21c) en un solo token, para que no se rompan en la
    tokenización por defecto de scikit-learn."""
    texto = RE_METRICA.sub(_proteger_metrica, texto)
    texto = RE_COMPUESTO.sub(_proteger_compuesto, texto)
    return texto


def _lematizar_palabra(palabra: str) -> str:
    if len(palabra) <= 4 or "_" in palabra:
        return palabra
    for suf in SUFIJOS_VERBALES:
        if palabra.endswith(suf) and len(palabra) - len(suf) >= 3:
            return palabra[: -len(suf)]
    return palabra


def lematizar_ligero(tokens: list[str]) -> list[str]:
    return [_lematizar_palabra(t) for t in tokens]


def fusionar_negaciones(tokens: list[str]) -> list[str]:
    """'no' nunca se elimina como stopword: se fusiona con el token
    siguiente (no_levanto, no_responde) para preservar la polaridad
    técnica negativa."""
    resultado = []
    i = 0
    while i < len(tokens):
        actual = tokens[i]
        if actual == "no" and i + 1 < len(tokens):
            resultado.append(f"no_{tokens[i + 1]}")
            i += 2
        else:
            resultado.append(actual)
            i += 1
    return resultado


def tokenizar(texto: str) -> list[str]:
    return [t for t in texto.split(" ") if t]


def preprocesar(texto: str) -> str:
    """Pipeline completo. Devuelve el texto listo para TF-IDF: limpio,
    normalizado, con compuestos protegidos, lematizado ligeramente y con
    negaciones fusionadas. Se entrega como string (tokens unidos por
    espacio) para poder pasarlo directo a TfidfVectorizer."""
    texto = limpiar_texto(texto)
    texto = proteger_tokens_compuestos(texto)
    texto = normalizar(texto)
    tokens = tokenizar(texto)
    tokens = lematizar_ligero(tokens)
    tokens = fusionar_negaciones(tokens)
    return " ".join(tokens)
