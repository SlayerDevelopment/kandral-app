FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Entrena los 3 clasificadores durante el build (dataset de 1000 filas,
# ~1s) y genera models/*.pkl, models/metrics.json y el dataset
# enriquecido. No requiere HF_TOKEN: el entrenamiento es 100% local.
RUN python train_models.py

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app"]
