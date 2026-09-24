"""
Chatbot orquestador: LLM externo (Hugging Face Inference Providers, API
compatible con OpenAI) con function calling sobre los 5 modelos ya
construidos.

El LLM nunca inventa cifras ni resultados: interpreta el mensaje del
usuario, decide qué función Python invocar (analizar texto, buscar en el
histórico, generar resumen de turno), y redacta la respuesta final en
lenguaje natural únicamente sobre los datos reales que esas funciones
devuelven.
"""
import json
import os

import pandas as pd
from openai import OpenAI

from nlp.preprocessing import preprocesar
from nlp.ner import extraer_entidades, resumir_por_tipo
from nlp.sentiment import analizar_reglas
from nlp.summarizer import generar_resumen

MODELO_LLM = os.environ.get("HF_CHAT_MODEL", "openai/gpt-oss-20b")

TURNOS = ["Mañana", "Tarde", "Noche"]
AREAS = ["Networking", "SysAdmin", "Storage & Backup", "Facilities"]
SEVERIDADES = ["P1 - Crítica", "P2 - Alta", "P3 - Media", "P4 - Baja"]
SENTIMIENTOS = ["Negativo/Crítico", "Neutro", "Positivo/Favorable"]

SYSTEM_PROMPT = f"""Sos el asistente de PLN de Kandral, un datacenter. Ayudás
al equipo de turno a analizar bitácoras nuevas y a consultar el histórico de
1000 registros de bitácora.

Tenés 3 herramientas:
- analizar_texto_bitacora: cuando el usuario pega un texto de bitácora nuevo
  para analizar.
- buscar_registros_historicos: cuando pregunta algo sobre el histórico
  (turnos: {TURNOS}; áreas: {AREAS}; severidades: {SEVERIDADES}).
- generar_resumen_turno: cuando pide un resumen o estado operacional de un
  turno completo.

Reglas estrictas:
- NUNCA inventes cifras, clasificaciones, tickets ni contenido de registros.
  Toda esa información sale exclusivamente de lo que te devuelven las
  herramientas.
- Si una pregunta requiere datos del dataset o de un análisis, SIEMPRE llamá
  a la herramienta correspondiente antes de responder; no respondas de
  memoria.
- Si el resultado de una herramienta está vacío o no alcanza para responder,
  decilo explícitamente en vez de rellenar con suposiciones.
- Respondé siempre en español, de forma clara y profesional, citando datos
  concretos (área, severidad, ticket, activo) cuando estén disponibles.
- Recordá que ningún resumen o análisis reemplaza la validación humana del
  supervisor de turno.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analizar_texto_bitacora",
            "description": (
                "Analiza un texto nuevo de bitácora que el usuario pegó en "
                "el chat y devuelve área técnica, severidad, sentimiento "
                "operacional (por reglas y por clasificador) y entidades "
                "detectadas (activos, ubicaciones, tickets, métricas)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "texto": {"type": "string", "description": "El texto de bitácora a analizar"},
                },
                "required": ["texto"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_registros_historicos",
            "description": (
                "Busca en los 1000 registros históricos del dataset de "
                "bitácoras de Kandral, filtrando por turno, área técnica, "
                "severidad, sentimiento y/o una palabra o frase en el texto."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "turno": {"type": "string", "enum": TURNOS},
                    "area_tecnica": {"type": "string", "enum": AREAS},
                    "severidad": {"type": "string", "enum": SEVERIDADES},
                    "sentimiento_regla": {"type": "string", "enum": SENTIMIENTOS},
                    "texto_contiene": {
                        "type": "string",
                        "description": "Palabra o frase que debe aparecer en el texto de la bitácora",
                    },
                    "limite": {
                        "type": "integer",
                        "description": "Cantidad máxima de registros a devolver (por defecto 10, máximo 15)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generar_resumen_turno",
            "description": (
                "Genera el resumen estructurado (estado operacional con "
                "semáforo, incidentes cerrados, tareas pendientes "
                "priorizadas por severidad) de todos los registros "
                "históricos de un turno completo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "turno": {"type": "string", "enum": TURNOS},
                },
                "required": ["turno"],
            },
        },
    },
]

COLUMNAS_REGISTRO = [
    "id", "turno", "area_tecnica", "severidad", "activo_principal",
    "ticket_asociado", "sentimiento_regla", "texto_bitacora",
]


class ChatbotKandral:
    def __init__(self, df: pd.DataFrame, modelos: dict):
        self.df = df
        self.modelos = modelos
        self.funciones = {
            "analizar_texto_bitacora": self._analizar_texto,
            "buscar_registros_historicos": self._buscar_registros,
            "generar_resumen_turno": self._resumen_turno,
        }
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError(
                    "Falta HF_TOKEN en las variables de entorno. "
                    "Agregalo a tu .env local (o como Secret en el despliegue)."
                )
            self._client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=token)
        return self._client

    # --- herramientas ---

    def _analizar_texto(self, texto: str = "") -> dict:
        texto = texto or ""
        texto_limpio = preprocesar(texto)
        entidades = extraer_entidades(texto)
        reglas = analizar_reglas(texto)
        return {
            "area_tecnica": self.modelos["area"].predict([texto_limpio])[0],
            "severidad": self.modelos["severidad"].predict([texto_limpio])[0],
            "sentimiento_reglas": reglas["etiqueta"],
            "sentimiento_clasificador": self.modelos["sentimiento"].predict([texto_limpio])[0],
            "entidades": resumir_por_tipo(entidades),
        }

    def _buscar_registros(self, turno=None, area_tecnica=None, severidad=None,
                           sentimiento_regla=None, texto_contiene=None, limite=10) -> dict:
        df = self.df
        if turno in TURNOS:
            df = df[df["turno"] == turno]
        if area_tecnica in AREAS:
            df = df[df["area_tecnica"] == area_tecnica]
        if severidad in SEVERIDADES:
            df = df[df["severidad"] == severidad]
        if sentimiento_regla in SENTIMIENTOS:
            df = df[df["sentimiento_regla"] == sentimiento_regla]
        if texto_contiene:
            df = df[df["texto_bitacora"].str.contains(str(texto_contiene), case=False, na=False)]

        try:
            limite = int(limite)
        except (TypeError, ValueError):
            limite = 10
        limite = max(1, min(limite, 15))

        total = len(df)
        muestra = df[COLUMNAS_REGISTRO].head(limite)
        muestra = muestra.astype(object).where(pd.notnull(muestra), None)

        return {
            "total_coincidencias": total,
            "registros_mostrados": muestra.to_dict(orient="records"),
        }

    def _resumen_turno(self, turno: str = "") -> dict:
        if turno not in TURNOS:
            return {"error": f"Turno inválido: '{turno}'. Debe ser uno de {TURNOS}."}

        subset = self.df[self.df["turno"] == turno][COLUMNAS_REGISTRO]
        subset = subset.astype(object).where(pd.notnull(subset), None)
        registros = subset.to_dict(orient="records")
        return generar_resumen(turno, registros)

    # --- orquestación ---

    def responder(self, mensaje: str, historial: list[dict] | None = None) -> dict:
        mensaje = (mensaje or "").strip()
        if not mensaje:
            return {"respuesta": "Escribime un mensaje para poder ayudarte.", "herramientas_usadas": []}

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turno_previo in (historial or []):
            if turno_previo.get("role") in ("user", "assistant") and turno_previo.get("content"):
                messages.append({"role": turno_previo["role"], "content": turno_previo["content"]})
        messages.append({"role": "user", "content": mensaje})

        respuesta = self.client.chat.completions.create(
            model=MODELO_LLM, messages=messages, tools=TOOLS, tool_choice="auto",
        )
        mensaje_respuesta = respuesta.choices[0].message
        herramientas_usadas = []

        if mensaje_respuesta.tool_calls:
            messages.append(mensaje_respuesta)
            for tool_call in mensaje_respuesta.tool_calls:
                nombre = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                funcion = self.funciones.get(nombre)
                if funcion is None:
                    resultado = {"error": f"Herramienta desconocida: {nombre}"}
                else:
                    try:
                        resultado = funcion(**args)
                    except TypeError as e:
                        resultado = {"error": f"Argumentos inválidos para {nombre}: {e}"}

                herramientas_usadas.append({"nombre": nombre, "argumentos": args, "resultado": resultado})
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": nombre,
                    "content": json.dumps(resultado, ensure_ascii=False, default=str),
                })

            final = self.client.chat.completions.create(
                model=MODELO_LLM, messages=messages, tools=TOOLS, tool_choice="none",
            )
            texto_final = final.choices[0].message.content
        else:
            texto_final = mensaje_respuesta.content

        return {"respuesta": texto_final, "herramientas_usadas": herramientas_usadas}
