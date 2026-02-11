FROM python:3.11-slim

WORKDIR /app

# Installa dipendenze sistema
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Copia requirements
COPY requirements.txt .

# Installa dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download del modello per velocizzare lo startup
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('gravitee-io/bert-small-toxicity'); \
    AutoModelForSequenceClassification.from_pretrained('gravitee-io/bert-small-toxicity')"

# Copia codice applicazione
COPY main.py .

# Esponi porta
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

# Avvia servizio
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5002"]
